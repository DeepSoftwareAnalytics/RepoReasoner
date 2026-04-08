import argparse
import json
import logging
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Tuple, List

import ast
try:
    import astunparse
except ImportError:
    print("Error: 'astunparse' library is not installed. Please run 'pip install astunparse'.")
    exit(1)

# --- Global Configuration ---
logger = logging.getLogger(__name__)
HOST_PYTHON_REPOS_DIR = Path("../python_repos")
HOST_REWRITES_DIR = Path("../output_rewrites")
BASE_DATA_DIR = Path("../data_collection_align")
CONTAINER_REPO_BASE_PATH = "/app/repo_to_process"
PYTEST_TIMEOUT_SECONDS = 120

# --- AST Injector ---
class AssertReplacer(ast.NodeTransformer):
    def __init__(self, target_path: List[str], ground_truth_expr: str, injection_code: str):
        self.target_path = target_path; self.ground_truth_expr = ground_truth_expr
        self.replacement_nodes = ast.parse(injection_code).body
        self.current_path = []; self.node_replaced = False; self.in_target_function = False
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self.current_path.append(node.name)
        if len(self.target_path) > 1 and self.target_path[:-1] == self.current_path: self.generic_visit(node)
        self.current_path.pop(); return node
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        path_to_check = self.current_path + [node.name]
        if path_to_check == self.target_path:
            self.in_target_function = True; self.generic_visit(node); self.in_target_function = False
        return node
    def visit_Assert(self, node: ast.Assert) -> Any:
        if not self.in_target_function or self.node_replaced: return node
        if isinstance(node.test, ast.Compare) and len(node.test.ops) == 1:
            right_side_expr = astunparse.unparse(node.test.comparators[0]).strip()
            if right_side_expr == self.ground_truth_expr:
                self.node_replaced = True; return self.replacement_nodes
        return node

# --- Core Logic Functions ---
def get_test_file_content(gt_item: Dict[str, Any]) -> str:
    reponame, testpath = gt_item['reponame'], gt_item['testpath']
    condition = gt_item['condition']
    if condition == 'original':
        source_file_path = HOST_PYTHON_REPOS_DIR / reponame / testpath
    else:
        path = Path(testpath)
        rewrite_filename = f"{path.stem}_agent_rewrite.py"
        source_file_path = HOST_REWRITES_DIR / reponame / rewrite_filename
    if not source_file_path.exists():
        raise FileNotFoundError(f"Test source file not found: {source_file_path}")
    return source_file_path.read_text(encoding='utf-8')

def generate_modified_content_ast(original_content: str, gt_item: Dict[str, Any]) -> str:
    ground_truth_expr = gt_item['ground_truth']
    target_path = [gt_item['classname'], gt_item['funcname']] if gt_item.get('classname') else [gt_item['funcname']]
    injection_code = f"""
# --- Injected code by ground_truth_collector ---
try:
    import json, sys
    _gt_value = ({ground_truth_expr})
    _serializable_value = {{"status": "success", "value": _gt_value, "type": str(type(_gt_value))}}
    try: output = json.dumps(_serializable_value)
    except (TypeError, OverflowError):
        _serializable_value['value'] = repr(_gt_value)
        output = json.dumps(_serializable_value)
    print("\\n---GT_START---"); print(output); print("---GT_END---")
    sys.stdout.flush() # Force flush buffer
except Exception as e:
    import traceback
    error_info = {{"status": "failure", "error": str(e), "traceback": traceback.format_exc()}}
    print("\\n---GT_START---"); print(json.dumps(error_info)); print("---GT_END---")
    sys.stdout.flush() # Force flush buffer
# --- End of injected code ---
"""
    try:
        tree = ast.parse(original_content)
        replacer = AssertReplacer(target_path, ground_truth_expr, injection_code.strip())
        modified_tree = replacer.visit(tree)
        if not replacer.node_replaced:
            raise ValueError(f"AST traversal completed, but no matching assert statement was found in function '{'::'.join(target_path)}'.")
        ast.fix_missing_locations(modified_tree)
        return astunparse.unparse(modified_tree)
    except Exception as e:
        logger.error(f"Failed to modify code using AST (Task ID: {gt_item['task_id']}): {e}")
        raise

def get_runtime_value_in_container(gt_item: Dict[str, Any], container_name: str) -> Dict:
    """Execute the modified code inside the container and retrieve runtime value via stdout."""
    try:
        original_content = get_test_file_content(gt_item)
        modified_content = generate_modified_content_ast(original_content, gt_item)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".py", encoding='utf-8') as tmp_file:
            tmp_file.write(modified_content)
            local_tmp_path = tmp_file.name

        testpath_in_repo = Path(gt_item['testpath'])
        temp_filename = f"{testpath_in_repo.stem}_gt_collector_tmp.py"
        temp_test_relative_path = testpath_in_repo.parent / temp_filename
        container_temp_test_full_path = Path(CONTAINER_REPO_BASE_PATH) / temp_test_relative_path
        
        container_path_str = container_temp_test_full_path.as_posix()
        subprocess.run(["docker", "cp", local_tmp_path, f"{container_name}:{container_path_str}"], check=True, capture_output=True)
        os.unlink(local_tmp_path)
        
        pytest_selector = "::".join([gt_item['classname'], gt_item['funcname']] if gt_item.get('classname') else [gt_item['funcname']])
        
        # --- FIX: Add -s flag to disable output capturing ---
        pytest_command = f"pytest -s --timeout={PYTEST_TIMEOUT_SECONDS} {container_path_str}::{pytest_selector}"
        
        exec_result = subprocess.run(
            ["docker", "exec", "--workdir", CONTAINER_REPO_BASE_PATH, container_name, "bash", "-c", pytest_command],
            capture_output=True, text=True, encoding='utf-8', errors='ignore')
        
        stdout = exec_result.stdout
        if "---GT_START---" in stdout and "---GT_END---" in stdout:
            result_str = stdout.split("---GT_START---")[1].split("---GT_END---")[0].strip()
            return json.loads(result_str)
        else:
            # If test fails early, we can still extract clues from stdout
            return {"status": "failure", "error": "Execution markers not found in output.", "details": f"Pytest returncode: {exec_result.returncode}\nPytest stdout:\n{stdout}\nPytest stderr:\n{exec_result.stderr}"}
    
    except Exception as e:
        logger.error(f"Unexpected error occurred when processing task {gt_item['task_id']}: {e}", exc_info=False)
        return {"status": "failure", "error": f"Unexpected exception: {str(e)}"}
    finally:
        if 'container_temp_test_full_path' in locals():
            subprocess.run(["docker", "exec", container_name, "rm", "-f", container_temp_test_full_path.as_posix()], capture_output=True)

# --- The main function and everything after remains unchanged ---
def main():
    parser = argparse.ArgumentParser(description="Retrieve runtime values of Ground Truth via execution.")
    parser.add_argument("--output_dir", type=str, default="../groundtruth_collection", help="Path to directory for saving output results.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set logging level.")
    args = parser.parse_args()

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')

    OUTPUT_DIR = Path(args.output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"All outputs will be saved to directory: '{OUTPUT_DIR}'")

    original_output_file = OUTPUT_DIR / "original.jsonl"
    rewrite_output_file = OUTPUT_DIR / "rewrite.jsonl"

    if original_output_file.exists(): original_output_file.unlink()
    if rewrite_output_file.exists(): rewrite_output_file.unlink()

    logging.info("===== Phase 1: Planning all tasks... =====")
    evaluation_plan = defaultdict(list)
    CONDITIONS = ["original", "rewrite"]
    for condition in CONDITIONS:
        data_dir_for_condition = BASE_DATA_DIR / condition
        if not data_dir_for_condition.exists():
            logging.warning(f"Data directory '{data_dir_for_condition}' does not exist, skipping.")
            continue
        for file_path in data_dir_for_condition.glob("**/*.jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    item['condition'] = condition
                    reponame = item['reponame']
                    evaluation_plan[reponame].append(item)
    
    total_repos = len(evaluation_plan)
    total_tasks = sum(len(tasks) for tasks in evaluation_plan.values())
    logging.info(f"Task planning complete. Found {total_repos} repositories and {total_tasks} tasks to process.")

    logging.info("===== Phase 2: Starting execution by repository order =====")
    processed_tasks = 0
    file_handles = {}
    try:
        file_handles['original'] = open(original_output_file, 'a', encoding='utf-8')
        file_handles['rewrite'] = open(rewrite_output_file, 'a', encoding='utf-8')

        for i, (reponame, tasks) in enumerate(sorted(evaluation_plan.items())):
            print("\n" + "#"*80 + f"\n#  ({i+1}/{total_repos}) Processing repository: {reponame.upper()}\n" + "#"*80)
            container_name = f"agent-run-{reponame}"
            try:
                logging.info(f"Starting container '{container_name}' for repository '{reponame}'...")
                subprocess.run(["docker", "start", container_name], check=True, capture_output=True)
                for task in tasks:
                    task_id = task['task_id']
                    condition = task['condition']
                    logging.info(f"--- Processing task: {task_id} (Condition: {condition}) ---")
                    result_data = get_runtime_value_in_container(task, container_name)
                    final_log = {"task_id": task_id, "reponame": reponame, "condition": condition, "original_ground_truth": task['ground_truth'], "runtime_ground_truth": result_data}
                    output_file_handle = file_handles[condition]
                    output_file_handle.write(json.dumps(final_log) + '\n')
                    processed_tasks += 1
                    logging.info(f"--- Task {task_id} completed, status: {result_data.get('status', 'unknown')} ---")
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to prepare Docker command for repository '{reponame}'! Skipping this repository.")
                logging.error(f"  Stderr: {e.stderr.decode('utf-8', 'ignore') if e.stderr else 'N/A'}")
            except Exception as e:
                logging.error(f"Unknown error occurred while processing repository '{reponame}': {e}", exc_info=True)
            finally:
                logging.info(f"Repository '{reponame}' processing finished, stopping container '{container_name}'...")
                subprocess.run(["docker", "stop", container_name], capture_output=True, check=False)
    finally:
        for handle in file_handles.values():
            if handle: handle.close()
    
    logging.info("\n" + "="*60 + "\nAll tasks completed!\n" + "="*60)
    logging.info(f"Processed {processed_tasks}/{total_tasks} tasks in total.")
    logging.info(f"Runtime Ground Truth values have been saved to directory: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()