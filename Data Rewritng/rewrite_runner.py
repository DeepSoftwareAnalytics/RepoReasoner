import argparse
import subprocess
import os
import sys
import logging
from pathlib import Path
import shutil 
import json 

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

DOCKER_IMAGE_NAME = "python-agent-tracer:latest"

def process_repo_rewrites(task_file: Path, args: argparse.Namespace):
    """
    Manages the Docker container to perform test rewriting for a single repository.
    """
    repo_name = task_file.stem
    container_name = f"agent-run-{repo_name}"
    
    # 1. Define the path to the report file that specifies which test files to rewrite.
    report_functions_path = Path("../output_results") / repo_name / "report_functions.jsonl"
    if not report_functions_path.is_file():
        logger.error(f"Report file not found at '{report_functions_path}'. Cannot determine which files to rewrite. Skipping repo '{repo_name}'.")
        return

    # 2. Read the report file and extract the set of unique test files to be rewritten.
    try:
        with open(report_functions_path, 'r', encoding='utf-8') as f:
            allowed_test_files = {json.loads(line)['test_file'] for line in f}
        logger.info(f"Found {len(allowed_test_files)} unique test files to rewrite for repo '{repo_name}' based on the report.")
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing report file '{report_functions_path}': {e}. Skipping repo '{repo_name}'.")
        return

    # 3. Read the original, unfiltered tasks from the file provided (e.g., from 'data_collection').
    try:
        with open(task_file, 'r', encoding='utf-8') as f:
            all_tasks = [json.loads(line) for line in f]
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing original task file '{task_file}': {e}. Skipping repo '{repo_name}'.")
        return

    # 4. Filter the tasks. Keep only those whose 'testpath' is in our set of allowed files.
    filtered_tasks = [
        task for task in all_tasks if task.get('testpath') in allowed_test_files
    ]

    if not filtered_tasks:
        logger.warning(f"After filtering, no rewrite tasks remain for repo '{repo_name}'. Skipping.")
        return
        
    logger.info(f"Filtered tasks. {len(filtered_tasks)} functions will be rewritten for repo '{repo_name}'.")

    # 5. Create a temporary task file to pass to the container.
    temp_task_file_path = task_file.parent / f"temp_filtered_{repo_name}.jsonl"
    try:
        with open(temp_task_file_path, 'w', encoding='utf-8') as f:
            for task in filtered_tasks:
                f.write(json.dumps(task) + '\n')
    except IOError as e:
        logger.error(f"Failed to write temporary filtered task file: {e}. Skipping repo '{repo_name}'.")
        return
        
    logger.info(f"--- Processing rewrites for repository: {repo_name} ---")

    try:
        check_proc = subprocess.run(["docker", "inspect", container_name], check=False, capture_output=True)
        if check_proc.returncode != 0:
            logger.error(f"Container '{container_name}' not found. Please run the initial test detection agent first to build and configure the container.")
            if temp_task_file_path.exists(): os.remove(temp_task_file_path) # Clean up temp file
            return

        logger.info(f"Starting existing container '{container_name}'...")
        subprocess.run(["docker", "start", container_name], check=True, capture_output=True)

        # 1. Copy the *filtered* task file into the container.
        container_task_path = "/app/rewrite_tasks.jsonl"
        logger.info(f"Copying filtered task file '{temp_task_file_path}' to '{container_name}:{container_task_path}'...")
        subprocess.run(
            ["docker", "cp", str(temp_task_file_path), f"{container_name}:{container_task_path}"],
            check=True
        )

        # 2. Reliably copy the agent code into the container.
        agent_code_path = Path("rewrite_agent")
        if not agent_code_path.is_dir() or not (agent_code_path / "run.py").exists():
            logger.error(f"Agent code directory '{agent_code_path}' seems incomplete or not found on host. Aborting.")
            return
            
        container_agent_path = "/app/rewrite_agent"
        logger.info(f"Copying agent code '{agent_code_path}' to '{container_name}:{container_agent_path}'...")
        
        subprocess.run(["docker", "exec", container_name, "rm", "-rf", container_agent_path], check=False)
        subprocess.run(["docker", "exec", container_name, "mkdir", "-p", container_agent_path], check=True)
        subprocess.run(
            ["docker", "cp", f"{str(agent_code_path)}/.", f"{container_name}:{container_agent_path}"],
            check=True
        )
        
        # # 3. Copy the latest runnable_agent_batch code into the container (overwrite existing)
        # runnable_agent_path = Path("runnable_agent_batch")
        # if not runnable_agent_path.is_dir() or not (runnable_agent_path / "run.py").exists():
        #     logger.error(f"Runnable agent code directory '{runnable_agent_path}' seems incomplete or not found on host. Aborting.")
        #     return
            
        # container_runnable_agent_path = "/app/runnable_agent_batch"
        # logger.info(f"Copying latest runnable_agent_batch code '{runnable_agent_path}' to '{container_name}:{container_runnable_agent_path}' (overwriting existing)...")
        
        # subprocess.run(
        #     ["docker", "cp", f"{str(runnable_agent_path)}/.", f"{container_name}:{container_runnable_agent_path}"],
        #     check=True
        # )
        
        # # 4. Copy the latest API configuration to the container
        # api_key_path = Path("API_KEY.txt")
        # if not api_key_path.exists():
        #     logger.error(f"API_KEY.txt not found at '{api_key_path}'. Aborting.")
        #     return
            
        # logger.info(f"Copying latest API configuration '{api_key_path}' to '{container_name}:/app/API_KEY.txt'...")
        # subprocess.run(
        #     ["docker", "cp", str(api_key_path), f"{container_name}:/app/API_KEY.txt"],
        #     check=True
        # )
        
        # 5. Execute the rewrite agent inside the container
        logger.info("Executing rewrite_agent inside the container...")
        exec_command = [
            "docker", "exec",
            "--workdir", "/app",
            container_name,
            "python", "-m", "rewrite_agent.run",
            "--model_name", args.model_name,
            "--repo_name", repo_name,
            "--max_debug_attempts", str(args.max_debug_attempts),
        ]
        
        process = subprocess.Popen(exec_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in iter(process.stdout.readline, ''):
            logger.info(f"[REWRITE-AGENT-LOG] {line.strip()}")
        process.wait()
        
        agent_exit_code = process.returncode
        logger.info(f"Rewrite agent finished with exit code: {agent_exit_code}")

        # 6. Copy results back
        output_base_dir = Path("../output_rewrites")
        output_repo_dir = output_base_dir / repo_name
        output_repo_dir.mkdir(parents=True, exist_ok=True)

        container_output_dir = "/app/modified_tests/"
        logger.info(f"Copying all rewritten tests from '{container_name}:{container_output_dir}'...")
        try:
            subprocess.run(
                ["docker", "cp", f"{container_name}:{container_output_dir}/.", str(output_repo_dir)],
                check=True, capture_output=True
            )
            logger.info(f"Successfully copied rewritten tests to '{output_repo_dir}'.")
        except subprocess.CalledProcessError:
            logger.warning(f"Could not copy any rewritten files from container. This is normal if the agent failed to create any valid rewrites.")

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode('utf-8', errors='replace') if e.stderr else 'N/A'
        logger.error(f"[FAILED] A docker command failed for repo '{repo_name}'. Stderr: {stderr_output}")
    except Exception as e:
        logger.error(f"An unexpected error occurred for repo '{repo_name}': {e}", exc_info=True)
    finally:
        logger.info(f"Stopping container '{container_name}'.")
        subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)
        if temp_task_file_path.exists():
            try:
                os.remove(temp_task_file_path)
                logger.info(f"Removed temporary task file '{temp_task_file_path}'.")
            except OSError as e:
                logger.warning(f"Could not remove temporary task file '{temp_task_file_path}': {e}")
        logger.info(f"--- Finished processing rewrites for: {repo_name} ---\n")

def main():
    parser = argparse.ArgumentParser(description="Test Rewriting Agent Runner")
    parser.add_argument('--tasks_dir', type=str, default='../data_collection', help="Directory containing the .jsonl files with rewrite tasks.")
    parser.add_argument('--model_name', type=str, default="gpt-4.1", help="Model name for rewriting and debugging.")
    parser.add_argument('--max_debug_attempts', type=int, default=5, help="Max debug attempts per test function.")
    

    parser.add_argument(
        '--repos', 
        nargs='+',  
        type=str, 
        default=None, 
        help="A list of specific repository names to process. If not provided, all repos in the tasks_dir will be processed."
    )


    args = parser.parse_args()

    try:
        if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            logger.error("[FAILED] Docker daemon is not running. Please start Docker.")
            sys.exit(1)
    except FileNotFoundError:
        logger.error("[FAILED] `docker` command not found. Is Docker installed and in your PATH?")
        sys.exit(1)

    tasks_path = Path(args.tasks_dir)
    if not tasks_path.is_dir():
        logger.error(f"Tasks directory not found: '{tasks_path}'")
        sys.exit(1)

    # Initially, find all possible task files.
    all_jsonl_files = sorted(list(tasks_path.glob("*.jsonl")))
    if not all_jsonl_files:
        logger.error(f"No .jsonl task files found in '{tasks_path}'")
        sys.exit(1)

    # 2. Filter the list of files based on the `--repos` argument.
    
    files_to_process = []
    if args.repos:
        # If the user specified a list of repos, filter `all_jsonl_files`.
        logger.info(f"Targeting {len(args.repos)} specific repository/repositories: {', '.join(args.repos)}")
        
        # Using a set for efficient lookup of requested repos.
        requested_repos_set = set(args.repos)
        files_to_process = [
            f for f in all_jsonl_files if f.stem in requested_repos_set
        ]
        
        # Provide feedback if some requested repos were not found.
        found_repos_set = {f.stem for f in files_to_process}
        missing_repos = requested_repos_set - found_repos_set
        if missing_repos:
            logger.warning(f"Could not find task files for the following requested repos: {', '.join(sorted(list(missing_repos)))}")
            
        if not files_to_process:
            logger.error(f"None of the requested repositories have corresponding .jsonl files in '{tasks_path}'. Exiting.")
            sys.exit(1)

    else:
        # If no specific repos were requested, process all found files.
        logger.info(f"No specific repos requested. Found {len(all_jsonl_files)} repositories to process.")
        files_to_process = all_jsonl_files

    logger.info(f"Starting processing for {len(files_to_process)} repositories.")
    
    # 3. Iterate over the potentially filtered list.
    for task_file in files_to_process:
        process_repo_rewrites(task_file, args)

    logger.info("All rewriting tasks complete.")

if __name__ == "__main__":
    os.environ['PYTHONUTF8'] = '1'
    main()