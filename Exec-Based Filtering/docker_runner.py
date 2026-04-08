# file: docker_runner.py (Fixed to prepend 'python_repos/')

import argparse
import subprocess
import os
import sys
import logging
from pathlib import Path
import shutil

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout)
        # logging.FileHandler(...) is removed from here
    ]
)
logger = logging.getLogger(__name__)

DOCKER_IMAGE_NAME = "python-agent-tracer:latest"

def process_repo_in_docker(repo_host_path: Path, args: argparse.Namespace):
    repo_name = repo_host_path.name
    container_name = f"agent-run-{repo_name}"
    should_delete_container = False  
    
    logger.info(f"--- Processing repository: {repo_name} ---")

    repo_container_path = "/app/repo_to_process"
    
    try:
        check_proc = subprocess.run(
            ["docker", "inspect", container_name],
            check=False, capture_output=True
        )
        container_exists = check_proc.returncode == 0

        if container_exists:
            logger.info(f"Found existing container '{container_name}'. Reusing it.")
            subprocess.run(["docker", "start", container_name], check=True, capture_output=True)
            
            logger.info(f"Removing old repo code from container and copying fresh version...")
            subprocess.run(
                ["docker", "exec", container_name, "rm", "-rf", repo_container_path],
                check=False, capture_output=True # check=False因为目录可能不存在
            )
            subprocess.run(
                ["docker", "cp", str(repo_host_path), f"{container_name}:{repo_container_path}"],
                check=True, capture_output=True
            )
        else:
            logger.info(f"No existing container found. Creating new container '{container_name}'...")
            subprocess.run(
                ["docker", "run", "-d", "--name", container_name, "--workdir", "/app", DOCKER_IMAGE_NAME, "tail", "-f", "/dev/null"],
                check=True, capture_output=True
            )
            logger.info(f"Copying '{repo_host_path}' to new container '{container_name}:{repo_container_path}'...")
            subprocess.run(["docker", "cp", str(repo_host_path), f"{container_name}:{repo_container_path}"], check=True, capture_output=True)

        api_key_path = "API_KEY.txt"
        if not os.path.exists(api_key_path):
            raise FileNotFoundError(f"'{api_key_path}' not found!")

        subprocess.run([
            "docker", "cp", 
            api_key_path,
            f"{container_name}:/app/{api_key_path}"
        ], check=True, capture_output=True)

        logger.info("Executing agent inside the container...")
        exec_command = [
            "docker", "exec", container_name,
            "python", "-m", "runnable_agent_batch.run",
            "--model_name", args.model_name,
            "--repo_path", repo_container_path,
            "--repo_name", repo_name,
            "--repo_language", "Python",
            "--max_iterations", str(args.max_iterations),
            "--timeout_per_command", str(args.timeout_per_command),
            "--timeout_pytest", str(args.timeout_pytest)
        ]
        
        process = subprocess.Popen(exec_command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in iter(process.stdout.readline, ''):
            logger.info(f"[AGENT-LOG] {line.strip()}")
        process.wait()
        
        agent_exit_code = process.returncode
        logger.info(f"Agent finished with exit code: {agent_exit_code}")
        
        # *** We use hard coding about output directory ***
        output_base_dir = Path("../output_results")
        output_repo_dir = output_base_dir / repo_name
        output_repo_dir.mkdir(parents=True, exist_ok=True)

        report_files_to_copy = ["report_files.jsonl", "report_functions.jsonl"]
        required_files_exist = True
        
        container_log_dir = f"/app/new_test_detection_logs/{repo_name}"

        logger.info(f"Copying all generated files from container '{container_name}:{container_log_dir}'...")
        try:
            subprocess.run(
                ["docker", "cp", f"{container_name}:{container_log_dir}/.", str(output_repo_dir)],
                check=True, capture_output=True
            )
            logger.info(f"Successfully copied agent-generated files to '{output_repo_dir}'.")
        except subprocess.CalledProcessError:
            logger.warning(f"Could not copy agent-generated files from container. This is normal if the agent failed before creating logs.")
        
        logger.info(f"Copying pytest reports from container repo root '{container_name}:{repo_container_path}'...")
        for report_file in report_files_to_copy:
            container_report_path = f"{container_name}:{repo_container_path}/{report_file}"
            try:
                subprocess.run(
                    ["docker", "cp", container_report_path, str(output_repo_dir)],
                    check=True, capture_output=True
                )
                logger.info(f"Successfully copied '{report_file}' to '{output_repo_dir}'.")
            except subprocess.CalledProcessError:
                logger.warning(f"Could not copy report file '{report_file}'. This might be expected if pytest did not complete successfully.")
                required_files_exist = False
        
        for report_file in report_files_to_copy:
            if not (output_repo_dir / report_file).exists():
                logger.warning(f"Required report file '{report_file}' was not generated successfully.")
                required_files_exist = False
        
        should_delete_container = False
        if agent_exit_code != 0:
            logger.warning(f"Agent failed with exit code {agent_exit_code}. Container will be deleted.")
            should_delete_container = True
        elif not required_files_exist:
            logger.warning(f"Required report files were not generated successfully. Container will be deleted.")
            should_delete_container = True
        else:
            logger.info(f"Agent completed successfully and all required files were generated. Container will be preserved for reuse.")

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.decode('utf-8', errors='replace') if e.stderr else 'N/A'
        logger.error(f"[FAILED] A docker command failed for repo '{repo_name}'. Stderr: {stderr_output}")
        should_delete_container = True  
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        should_delete_container = True  
    finally:
        if should_delete_container:
            logger.info(f"Deleting container '{container_name}' due to failure or missing required files.")
            subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)
            subprocess.run(["docker", "rm", container_name], check=False, capture_output=True)
        else:
            logger.info(f"Stopping container '{container_name}' to conserve resources. It will be reused later.")
            subprocess.run(["docker", "stop", container_name], check=False, capture_output=True)
        logger.info(f"--- Finished processing repository: {repo_name} ---\n")


def main():
    parser = argparse.ArgumentParser(description="Docker-based Test and Trace Agent Runner (uses pre-built image)")
    parser.add_argument('--repos_list', type=str, default='../python_repos/python_repos.txt', help="File with list of repo names.")
    parser.add_argument('--repos_base_dir', type=str, default='../python_repos', help="The base directory where repositories are stored.")
    parser.add_argument('--model_name', type=str, default="gpt-4.1", help="Model name.")
    parser.add_argument('--max_iterations', type=int, default=15, help="Max iterations per repo.")
    parser.add_argument('--timeout_per_command', type=int, default=600, help="Default timeout per command (seconds).")
    parser.add_argument('--timeout_pytest', type=int, default=3600, help="Special timeout for pytest execution (seconds).")
    args = parser.parse_args()

    try:
        if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
            logger.error("[FAILED] Docker daemon is not running. Please start Docker.")
            sys.exit(1)
    except FileNotFoundError:
        logger.error("[FAILED] `docker` command not found. Is Docker installed and in your PATH?")
        sys.exit(1)
        
    image_exists_proc = subprocess.run(["docker", "images", "-q", DOCKER_IMAGE_NAME], capture_output=True, text=True)
    if not image_exists_proc.stdout.strip():
        logger.error(f"[FAILED] The required Docker image '{DOCKER_IMAGE_NAME}' was not found locally.")
        logger.error("Please build it manually first by running: docker build -t python-agent-tracer:latest .")
        sys.exit(1)
    
    logger.info(f"Found local image '{DOCKER_IMAGE_NAME}'. Starting the process...")

    repos_list_file = Path(args.repos_list)
    if not repos_list_file.exists():
        logger.error(f"Repository list file not found: '{repos_list_file}'")
        sys.exit(1)
    
    repos_base_path = Path(args.repos_base_dir)
    if not repos_base_path.exists() or not repos_base_path.is_dir():
        logger.error(f"Repositories base directory not found: '{repos_base_path}'")
        sys.exit(1)

    with open(repos_list_file, 'r', encoding='utf-8') as f:
        repo_names = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    logger.info(f"Found {len(repo_names)} repository names to process from base directory '{repos_base_path}'.")
    
    root_logger = logging.getLogger()
    file_handler = None 

    for repo_name in repo_names:
        if file_handler:
            root_logger.removeHandler(file_handler)

        # *** We use hard coding about output directory ***
        output_repo_dir = Path("../output_results") / repo_name
        output_repo_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = output_repo_dir / "docker_runner_main.log"


        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s', '%m/%d/%Y %H:%M:%S')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

        repo_path = repos_base_path / repo_name
        
        if repo_path.exists() and repo_path.is_dir():
            process_repo_in_docker(repo_path, args)
        else:
            logger.warning(f"[SKIPPING] Constructed path not found or is not a directory: {repo_path}")

    if file_handler:
        root_logger.removeHandler(file_handler)
if __name__ == "__main__":
    os.environ['PYTHONUTF8'] = '1'
    main()