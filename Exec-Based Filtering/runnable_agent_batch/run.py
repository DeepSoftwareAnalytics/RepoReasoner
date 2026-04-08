import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import signal

import tiktoken
from runnable_agent_batch.generator import Generator
from runnable_agent_batch.prompts.system_prompts import get_test_detection_prompt_for_language
from runnable_agent_batch.prompts.user_prompts import (
    generate_initial_user_prompt,
    generate_followup_user_prompt,
    CONFTEST_PY_CONTENT
)
from runnable_agent_batch.utils import get_project_structure, truncate_content


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class TestDetectionPipeline:
    def __init__(self, args, encoding):
        self.args = args
        self.encoding = encoding
        self.generator = None
        
        self.repo_path = Path(args.repo_path).resolve()
        self.repo_language = args.repo_language
        self.repo_name = args.repo_name 

        if not self.repo_path.exists():
            raise FileNotFoundError(f"Repository path does not exist inside container: {self.repo_path}")
        
        self.max_iterations = args.max_iterations
        self.timeout_default = args.timeout_per_command
        self.timeout_pytest = args.timeout_pytest
        self.current_iteration = 0
        self.execution_history = []
        self.created_files = []
        self.phase = "SETUP"
        
        self.system_prompt = get_test_detection_prompt_for_language(self.repo_language)
        
        self.output_dir = Path("new_test_detection_logs") / self.repo_name
        self.setup_output_directory()
        
        self.initialize_generator()

    def setup_output_directory(self):
        """Setup output directory for logs."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def initialize_generator(self):
        """Initialize the generator with the system prompt and a dedicated log path."""
        conversation_log_file = self.output_dir / "model_conversation.jsonl"
        self.generator = Generator(
            self.args, 
            logger, 
            self.system_prompt,
            conversation_log_path=str(conversation_log_file)
        )

    def get_current_project_structure(self) -> str:
        """Get current project structure from the working directory."""
        return get_project_structure(self.repo_path)

    def extract_file_blocks(self, text: str) -> List[Dict[str, str]]:
        """Extract file creation blocks from model response."""
        file_pattern = r'```file:([^\n]+)\n(.*?)```'
        matches = re.findall(file_pattern, text, re.DOTALL)
        
        files = []
        for filename, content in matches:
            filename = filename.strip()
            content = content.strip()
            files.append({
                'filename': filename,
                'content': content
            })
            logger.info(f"Found file block: {filename}")
        
        return files

    def create_files(self, files: List[Dict[str, str]]) -> Tuple[bool, str]:
        """Create files from file blocks, with special handling for conftest.py."""
        if not files:
            return True, "No files to create"
        
        results = []
        all_success = True
        
        for file_info in files:
            filename = file_info['filename']
            content = file_info['content']
            
            try:
                # Special handling to inject full conftest.py content
                if 'conftest.py' in filename:
                    # Heuristic to check if the model provided a shorthand version
                    if 'import hunter' not in content:
                        logger.info("Model provided a shorthand for conftest.py. Expanding to full content.")
                        # Assumes content is the `SOURCE_DIRS_TO_TRACK = [...]` line
                        source_line = content 
                        content = CONFTEST_PY_CONTENT.replace(
                            "SOURCE_DIRS_TO_TRACK = ['<placeholder_source_dir>']", source_line
                        )
                
                file_path = self.repo_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                if filename.endswith('.sh'):
                    os.chmod(file_path, 0o755)
                
                self.created_files.append(str(file_path))
                results.append(f"✅ Created file: {filename}")
                logger.info(f"Created file: {file_path}")
            except Exception as e:
                results.append(f"❌ Failed to create {filename}: {str(e)}")
                logger.error(f"Failed to create file {filename}: {e}")
                all_success = False
        
        return all_success, "\n".join(results)

    def extract_bash_commands(self, text: str) -> List[str]:
        """Extract bash commands from model response."""        
        bash_pattern = r'```bash\s*\n(.*?)\n```'
        matches = re.findall(bash_pattern, text, re.DOTALL)
        
        commands = []
        for match in matches:
            lines = [line.strip() for line in match.split('\n')]
            lines = [line for line in lines if line and not line.startswith('#')]
            commands.extend(lines)
        
        return commands

    def check_status(self, text: str) -> Optional[str]:
        """Check for status markers in the response."""
        if re.search(r'```status\s*\n\s*failed\s*\n```', text, re.IGNORECASE):
            return 'failed'
        if re.search(r'```status\s*\n\s*success\s*\n```', text, re.IGNORECASE):
            return 'success'
        return None

    def execute_command(self, command: str) -> Tuple[bool, str, str]:
        """Execute a single command, using a special timeout for pytest."""
        
        if 'pytest' in command:
            timeout = self.timeout_pytest
            logger.info(f"Using PYTEST timeout ({timeout}s) for command: {command}")
        else:
            timeout = self.timeout_default
        
        try:
            logger.info(f"Executing command in '{self.repo_path}': {command}")
            process = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=self.repo_path,
                timeout=timeout, 
                encoding='utf-8',
                errors='replace'
            )
            success = process.returncode == 0
            return success, process.stdout, process.stderr
        except subprocess.TimeoutExpired:
            error_msg = f"Command timed out after {timeout} seconds" 
            logger.warning(error_msg)
            return False, "", error_msg
        except Exception as e:
            error_msg = f"Failed to execute command: {str(e)}"
            logger.error(error_msg)
            return False, "", error_msg

    def execute_commands(self, commands: List[str]) -> Tuple[str, List[dict]]:
        """Execute commands and return results."""
        if not commands:
            return "No commands found in the response.", []
        
        logger.info(f"Executing {len(commands)} command(s)...")
        results = []
        execution_details = []
        
        for i, command in enumerate(commands, 1):
            start_time = time.time()
            success, stdout, stderr = self.execute_command(command)
            execution_time = time.time() - start_time
            
            detail = {
                'command': command,
                'success': success,
                'stdout': stdout,
                'stderr': stderr,
                'execution_time': execution_time
            }
            execution_details.append(detail)
            
            result_text = f"--- Command: {command} ---\nSUCCESS: {success}\n"
            if stdout:
                result_text += f"STDOUT:\n{truncate_content(stdout, 2000, self.encoding)}\n"
            if stderr:
                result_text += f"STDERR:\n{truncate_content(stderr, 2000, self.encoding)}\n"
            results.append(result_text)
        
        return "\n".join(results), execution_details

    def save_iteration_files(self, iteration: int, user_input: str, model_output: str, 
                           execution_details: List[dict]):
        """Save detailed files for each iteration."""
        iteration_dir = self.output_dir / f"iteration_{iteration:02d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        
        with open(iteration_dir / "user_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(user_input)
        
        with open(iteration_dir / "model_response.txt", 'w', encoding='utf-8') as f:
            f.write(model_output)
            
        with open(iteration_dir / "execution_log.json", 'w', encoding='utf-8') as f:
            json.dump(execution_details, f, indent=2, ensure_ascii=False)


    def check_phase_completion(self, execution_details: List[dict]):
        """Check if the current phase's goal is met and transition if so."""
        if self.phase == "SETUP":
            for detail in execution_details:
                if 'pytest' not in detail['command']:
                    continue

                stdout = detail.get('stdout', '')
                stderr = detail.get('stderr', '')
                output = stdout + stderr
                
            
                crash_signal = "INTERNALERROR" in stderr or \
                               "ImportError" in stderr or \
                               "ModuleNotFoundError" in stderr or \
                               "ERROR collecting" in output

                success_signal = re.search(r'collected \d+ items', output) or \
                                 re.search(r'=\s*(\d+ passed|\d+ failed|\d+ skipped|no tests ran)', output)


                if success_signal and not crash_signal:
                    logger.info("✅ Phase 1 (SETUP) complete: `pytest` is now executable without collection errors.")
                    self.phase = "TRACING"
                    return True 
                else:
                    logger.warning("Phase 1 (SETUP) not yet complete. Pytest reported collection errors or did not run successfully.")

        return False

    def run(self):
        """Main pipeline execution with phase management."""
        user_prompt = generate_initial_user_prompt(self.repo_path, self.repo_language, self.encoding)
        final_status = 'unknown'

        while self.current_iteration < self.max_iterations:
            self.current_iteration += 1
            logger.info(f"=== Starting Iteration {self.current_iteration}/{self.max_iterations} [Phase: {self.phase}] ===")
            
            self.generator.slide_window_conversation()
            self.generator.messages.append({
                "role": "user",
                "content": user_prompt
            })

            repo_log_name = f"{self.repo_path.name}"
            response = self.generator.get_response(repo_name=self.repo_name)

            if response == -1:
                logger.error("Failed to get response from model. Stopping.")
                final_status = 'failed_model_response'
                break

            commands = self.extract_bash_commands(response)
            files = self.extract_file_blocks(response)
            status = self.check_status(response)

            _, file_results = self.create_files(files)
            execution_results, execution_details = self.execute_commands(commands)
            
            self.save_iteration_files(self.current_iteration, user_prompt, response, execution_details)

            # Check for explicit termination signals from the model
            if status == 'success':
                logger.info("🎉 Model reported final success. Terminating.")
                final_status = 'success'
                break
            if status == 'failed':
                logger.info("❌ Model reported failure. Terminating.")
                final_status = 'failed'
                break
            
            # Check for automatic phase transition
            if self.phase == "SETUP":
                self.check_phase_completion(execution_details)

            # Generate the next prompt based on the current (possibly updated) phase
            user_prompt = generate_followup_user_prompt(
                repo_path=self.repo_path, 
                language=self.repo_language, 
                phase=self.phase,
                execution_results=execution_results, 
                file_results=file_results,
                encoding=self.encoding
            )
        
        if self.current_iteration >= self.max_iterations:
            logger.warning(f"Reached maximum iterations ({self.max_iterations}).")
            final_status = 'max_iterations_reached'

        logger.info(f"🏁 Pipeline finished for {self.repo_path.name} with status: {final_status}")
        return final_status


def main():
    """Entry point when the script is run inside the Docker container."""
    parser = argparse.ArgumentParser(description="Test and Trace Agent (Containerized)")
    parser.add_argument('--model_name', type=str, required=True, help="Model name to use.")
    parser.add_argument('--repo_path', type=str, required=True, help="Path to the repository inside the container.")
    parser.add_argument('--repo_name', type=str, required=True, help="Actual name of the repository.")
    parser.add_argument('--repo_language', type=str, required=True, help="Language of the repository.")                   
    parser.add_argument('--max_iterations', type=int, required=True, help="Maximum number of iterations.")
    parser.add_argument('--timeout_per_command', type=int, required=True, help="Timeout per command in seconds.")
    parser.add_argument('--timeout_pytest', type=int, required=True, help="Special timeout for pytest execution in seconds.")
    args = parser.parse_args()

    logger.info("--- Starting Agent inside Container ---")
    logger.info(f"Repository Name: {args.repo_name}")
    logger.info(f"Repository: {args.repo_path}")
    logger.info(f"Language: {args.repo_language}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Max Iterations: {args.max_iterations}")
    logger.info(f"Command Timeouts (Default/Pytest): {args.timeout_per_command}s / {args.timeout_pytest}s")
    
    try:
        encoding = tiktoken.encoding_for_model("gpt-4")
        pipeline = TestDetectionPipeline(args, encoding)
        final_status = pipeline.run()
        
        # Exit with a code that the `docker_runner.py` can interpret
        if final_status == 'success':
            logger.info("✅ Agent completed successfully.")
            sys.exit(0)
        else:
            logger.info(f"❌ Agent finished with non-success status: {final_status}")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"💥 Agent pipeline failed with a critical error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(2)


if __name__ == "__main__":
    main()