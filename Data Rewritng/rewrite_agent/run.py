import argparse
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

# Assumes the generator.py from your original project is available in the python path.
from runnable_agent_batch.generator import Generator 
from .prompts import (
    REWRITE_SYSTEM_PROMPT, 
    generate_initial_file_rewrite_prompt, 
    generate_debug_prompt
)

# Import conftest tracking disable utility
try:
    from conftest_disable_tracking import find_and_disable_conftest_files, restore_all_conftest_files
except ImportError:
    # Fallback if the utility is not available
    def find_and_disable_conftest_files(repo_path):
        return []
    def restore_all_conftest_files(modified_files):
        pass

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class TestRewritePipeline:
    def __init__(self, args):
        self.args = args
        self.repo_name = args.repo_name
        self.repo_path = Path("/app/repo_to_process").resolve() # Hardcoded container path
        self.max_debug_attempts = args.max_debug_attempts
        
        # This is the "archive" directory for successfully rewritten files
        self.output_dir = Path("/app/modified_tests/")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the generator with the system prompt for rewriting
        self.generator = Generator(args, logger, REWRITE_SYSTEM_PROMPT)

    def extract_file_block(self, text: str) -> Optional[Dict[str, str]]:
        """Extracts the first file block from the model's response."""
        file_pattern = r'```file:([^\n]+)\n(.*?)```'
        match = re.search(file_pattern, text, re.DOTALL)
        if match:
            filename = match.group(1).strip()
            content = match.group(2).strip()
            logger.info(f"Found file block for: {filename}")
            return {'filename': filename, 'content': content}
        logger.warning("No file block found in the model response.")
        return None

    def execute_pytest(self, test_file_path: Path) -> Tuple[bool, str]:
        """Executes pytest on a single file and returns the result.
        Disables custom call chain tracking plugins during rewrite phase."""
        if not test_file_path.exists():
            return False, f"Error: Test file '{test_file_path}' does not exist."
        
        modified_conftest_files = find_and_disable_conftest_files(self.repo_path)
        
        try:
            command = f"pytest {str(test_file_path)} -p no:warnings --tb=short"
            logger.info(f"Executing pytest command: {command}")
            
            process = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                cwd=self.repo_path, timeout=300, encoding='utf-8', errors='replace'
            )
            success = process.returncode == 0
            output = process.stdout + "\n" + process.stderr
            return success, output
            
        except subprocess.TimeoutExpired:
            return False, "Pytest command timed out after 300 seconds."
        except Exception as e:
            return False, f"Failed to execute pytest: {e}"
        finally:
            if modified_conftest_files:
                restore_all_conftest_files(modified_conftest_files)
                logger.info(f"Restored {len(modified_conftest_files)} conftest.py files")

    def process_file_rewrite(self, test_path_str: str, tasks_in_file: List[dict]):
        """Manages the rewrite-test-debug loop for a single test file."""
        func_names_to_rewrite = [task['funcname'] for task in tasks_in_file]
        logger.info(f"--- Starting to process file: {test_path_str}, containing {len(tasks_in_file)} functions to rewrite ---")
        
        original_test_path = self.repo_path / test_path_str
        if not original_test_path.exists():
            logger.error(f"[SKIPPING] Original file not found: {original_test_path}")
            return

        try:
            original_content = original_test_path.read_text(encoding='utf-8')
        except Exception as e:
            logger.error(f"[SKIPPING] Could not read original file {original_test_path}: {e}")
            return
            
        # --- START OF FIX ---
        # Initialize the prompt for the first attempt.
        # This variable will be updated with a debug prompt in case of failure.
        current_prompt = generate_initial_file_rewrite_prompt(
            test_path_str, func_names_to_rewrite, original_content
        )
        # --- END OF FIX ---
        
        generated_code = ""
        new_test_filepath = None

        # Start the file-level debug loop
        for attempt in range(self.max_debug_attempts + 1): # +1 to allow initial attempt + max_debug_attempts
            if attempt > 0:
                logger.info(f"Debug Attempt {attempt}/{self.max_debug_attempts} for file {test_path_str}")
            else:
                logger.info(f"Initial rewrite attempt for file {test_path_str}")

            # --- START OF FIX ---
            # KEY CHANGE: Reset the conversation history for each attempt.
            # This prevents the message list from growing indefinitely and causing an OOM error.
            # The system prompt is always the first message.
            self.generator.init_conversation() 
            self.generator.messages.append({"role": "user", "content": current_prompt})
            # --- END OF FIX ---

            response = self.generator.get_response(repo_name=self.repo_name)
            
            if response == -1:
                logger.error("Failed to get response from model. Aborting task for this file.")
                # We can't generate a debug prompt without a response, so we must exit.
                return

            file_block = self.extract_file_block(response)
            if not file_block:
                # If we can't parse the file, our only option is to try again with a generic error.
                # Or, more simply, update the prompt to ask the model to fix its format.
                logger.error("Could not extract file block from response. Preparing debug prompt to ask for correction.")
                current_prompt = f"Your previous response was malformed. You MUST output the code in a single ```file:...``` block. Please try again.\n\nHere is the original request again:\n\n{current_prompt}"
                continue # Go to the next attempt

            generated_code = file_block['content']
            new_test_filename = file_block['filename']
            # Write to the temporary workspace within the project structure
            new_test_filepath = self.repo_path / new_test_filename
            
            new_test_filepath.parent.mkdir(parents=True, exist_ok=True)
            new_test_filepath.write_text(generated_code, encoding='utf-8')

            # Execute pytest on this single, newly created file
            success, pytest_output = self.execute_pytest(new_test_filepath)

            if success:
                logger.info(f"✅ SUCCESS: All tests in file '{new_test_filepath.name}' passed!")
                # Copy the successful file to the final archive directory
                final_output_path = self.output_dir / new_test_filepath.name
                final_output_path.write_text(generated_code, encoding='utf-8')
                logger.info(f"Copied successful file to {final_output_path}")
                return # End processing for this file

            # If failed, prepare the debug prompt for the *next* loop iteration
            logger.warning("File test failed. Preparing debug prompt for the next attempt.")
            logger.info(f"Pytest output:\n{pytest_output}")
            
            # --- START OF FIX ---
            # The new prompt for the next iteration is the debug prompt.
            # This prompt is self-contained and has all the necessary context.
            current_prompt = generate_debug_prompt(new_test_filename, generated_code, pytest_output)
            # --- END OF FIX ---
        
        logger.error(f"❌ FAILED: Max debug attempts reached for file {test_path_str}.")

        
    def run(self):
        """Main entry point for the agent."""
        task_file = Path("/app/rewrite_tasks.jsonl")
        if not task_file.exists():
            logger.critical("Task file '/app/rewrite_tasks.jsonl' not found inside container.")
            sys.exit(1)
        
        with open(task_file, 'r', encoding='utf-8') as f:
            tasks = [json.loads(line) for line in f]
            
        # Group tasks by test file path
        tasks_by_file = defaultdict(list)
        for task in tasks:
            tasks_by_file[task['testpath']].append(task)
        
        logger.info(f"Found {len(tasks)} functions to rewrite across {len(tasks_by_file)} files.")
        
        # Iterate through the grouped files, not the raw tasks
        for test_path_str, tasks_in_file in tasks_by_file.items():
            self.process_file_rewrite(test_path_str, tasks_in_file)
            
        logger.info(f"All rewrite tasks for repository '{self.repo_name}' have been processed.")


def main():
    parser = argparse.ArgumentParser(description="Test Rewriting Agent (Containerized)")
    parser.add_argument('--model_name', type=str, required=True, help="Model name.")
    parser.add_argument('--repo_name', type=str, required=True, help="Name of the repository.")
    parser.add_argument('--max_debug_attempts', type=int, default=5, help="Max debug attempts per test file.")
    # A dummy argument to satisfy the Generator's needs, even if not directly used here
    parser.add_argument('--repo_path', type=str, default="", help="Dummy repo path.")
    
    args = parser.parse_args()

    try:
        pipeline = TestRewritePipeline(args)
        pipeline.run()
        logger.info("✅ Rewrite agent completed successfully.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Rewrite agent pipeline failed with a critical error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        sys.exit(1)

if __name__ == "__main__":
    main()