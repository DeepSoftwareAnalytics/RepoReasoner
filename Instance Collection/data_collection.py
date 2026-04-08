import ast
import os
import json
import copy
import sys
import astunparse
from data_quality_filter import create_quality_filter

class TestFinder(ast.NodeVisitor):
    def __init__(self):
        self.found_tests = []
        self._current_class_name = None

    def visit_ClassDef(self, node: ast.ClassDef):
        original_class_name = self._current_class_name
        self._current_class_name = node.name
        self.generic_visit(node)
        self._current_class_name = original_class_name

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if node.name.startswith("test_"):
            self.found_tests.append({
                "node": node,
                "classname": self._current_class_name  
            })

        self.generic_visit(node)


class TaskIdGenerator:
    def __init__(self):
        self.current_id = 0
    
    def next(self) -> int:
        task_id = self.current_id
        self.current_id += 1
        return task_id

class AssertionTransformer(ast.NodeTransformer):
    def __init__(self, target_lineno: int, target_col_offset: int):
        self.target_lineno = target_lineno
        self.target_col_offset = target_col_offset
        self.transformed = False
        self.ground_truth = None

    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        if not (node.lineno == self.target_lineno and node.col_offset == self.target_col_offset):
            return node
        if not (isinstance(node.test, ast.Compare) and
                node.test.ops and isinstance(node.test.ops[0], ast.Eq)):
            return node
        original_node = node.test.comparators[0]
        self.ground_truth = astunparse.unparse(original_node).strip()
        self.transformed = True
        try:
            node.test.comparators[0] = ast.Constant(value='???', kind=None)
        except AttributeError:
            node.test.comparators[0] = ast.Name(id='???', ctx=ast.Load())
        return node

def process_test_file(file_path: str, repo_root: str, task_id_gen: TaskIdGenerator):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
        tree = ast.parse(source_code)
    except (SyntaxError, UnicodeDecodeError, PermissionError, FileNotFoundError) as e:
        print(f"warning: can not analyze {file_path}. reason: {e}. skip now", file=sys.stderr)
        return []

    imports_list = [astunparse.unparse(node).strip() for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    all_data_entries = []
    reponame = os.path.basename(repo_root)
    testpath = os.path.relpath(file_path, repo_root)
    testname = os.path.basename(file_path)
    
    quality_filter = create_quality_filter() if create_quality_filter else None

    finder = TestFinder()
    finder.visit(tree)

    for test_info in finder.found_tests:
        node = test_info["node"]
        classname = test_info["classname"] 

        if quality_filter:
            quality_asserts = quality_filter.filter_test_function(node)
            candidate_asserts = [item["assert_node"] for item in quality_asserts]
        else:
            candidate_asserts = [sn for sn in ast.walk(node) if isinstance(sn, ast.Assert) and isinstance(sn.test, ast.Compare) and sn.test.ops and isinstance(sn.test.ops[0], ast.Eq)]
        
        for assert_to_mask in candidate_asserts:
            func_copy = copy.deepcopy(node)
            transformer = AssertionTransformer(assert_to_mask.lineno, assert_to_mask.col_offset)
            masked_function_node = transformer.visit(func_copy)
            
            if transformer.transformed and transformer.ground_truth is not None:
                task_id_str = f"{reponame}_{task_id_gen.next()}"
                quality_analysis = quality_filter.analyze_assertion_complexity(assert_to_mask) if quality_filter else None
                
                data_entry = {
                    "task_id": task_id_str,
                    "reponame": reponame,
                    "testpath": testpath.replace('\\', '/'),
                    "testname": testname,
                    "classname": classname, 
                    "funcname": node.name,
                    "imports": imports_list,
                    "code": astunparse.unparse(node).strip(),
                    "masked_code": astunparse.unparse(masked_function_node).strip(),
                    "ground_truth": transformer.ground_truth
                }
                
                if quality_analysis:
                    data_entry["quality_analysis"] = quality_analysis
                
                all_data_entries.append(data_entry)
                
    return all_data_entries

def process_single_repo(repo_path: str, output_file: str):
    repo_abs_path = os.path.abspath(repo_path)
    if not os.path.isdir(repo_abs_path):
        print(f"error: '{repo_path}' does not exist. skip now.", file=sys.stderr)
        return
    task_id_gen = TaskIdGenerator()
    print(f"\n--- Repo: {os.path.basename(repo_abs_path)} ---")
    all_processed_data = []
    excluded_dirs = {'.venv', 'venv', '.git', 'node_modules', 'build', 'dist', '__pycache__'}
    for root, dirs, files in os.walk(repo_abs_path):
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        for file in files:
            if file.startswith("test_") and file.endswith(".py"):
                file_path = os.path.join(root, file)
                processed_data = process_test_file(file_path, repo_abs_path, task_id_gen)
                all_processed_data.extend(processed_data)
    if not all_processed_data:
        print(f"{os.path.basename(repo_abs_path)} can not find assert.")
        return
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_processed_data:
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"✅ success! {len(all_processed_data)} assert. save to: {output_file}")

def process_multiple_repos(repo_paths: list, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"all output save to: {os.path.abspath(output_dir)}")
    for repo_path in repo_paths:
        reponame = os.path.basename(os.path.abspath(repo_path))
        output_file = os.path.join(output_dir, f"{reponame}.jsonl")
        process_single_repo(repo_path, output_file)

def main():
    current_dir = os.getcwd()
    upper_dir = os.path.dirname(current_dir)
    python_repos_dir = os.path.join(upper_dir, 'python_repos')
    output_results_dir = os.path.join(upper_dir, 'output_results')
    data_collection_dir = os.path.join(upper_dir, 'data_collection')

    print("--- Stage 1: Start filtering repositories to process ---")
    if not os.path.isdir(output_results_dir):
        print(f"Error: Report directory '{output_results_dir}' not found. Script aborted.", file=sys.stderr)
        return
    if not os.path.isdir(python_repos_dir):
        print(f"Error: Source code directory '{python_repos_dir}' not found. Script aborted.", file=sys.stderr)
        return
    filtered_repo_names = []
    repo_subdirs = [d for d in os.listdir(output_results_dir) if os.path.isdir(os.path.join(output_results_dir, d))]
    for reponame in repo_subdirs:
        report_file = os.path.join(output_results_dir, reponame, 'report_functions.jsonl')
        if os.path.exists(report_file):
            if os.path.isdir(os.path.join(python_repos_dir, reponame)):
                filtered_repo_names.append(reponame)
                print(f"Found report '{report_file}', adding repository '{reponame}' to processing list.")
            else:
                print(f"Warning: Report file for repository '{reponame}' found, but source code not found in 'python_repos' directory. Skipped.")
        else:
            print(f"Info: No report file found in repository '{reponame}', skipping.")
    if not filtered_repo_names:
        print("\nNo repositories meeting the conditions were found (i.e., both report file and source code exist). Script exiting.")
        return
    print(f"\n✅ Filtering complete! Found {len(filtered_repo_names)} repositories ready for data extraction.")
    print("\n--- Stage 2: Start extracting assertion data from filtered repositories ---")
    repo_paths_to_process = [os.path.join(python_repos_dir, name) for name in filtered_repo_names]
    process_multiple_repos(repo_paths_to_process, data_collection_dir)
    print("\n--- All tasks completed ---")

if __name__ == '__main__':
    main()