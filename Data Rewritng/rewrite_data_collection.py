import ast
import os
import json
import astunparse
import copy
import sys
from collections import defaultdict
from typing import Union, Dict, List

# Import data quality filter
try:
    from data_quality_filter import create_quality_filter
except ImportError:
    print("Warning: Data quality filter module not found, will use basic filtering.", file=sys.stderr)
    create_quality_filter = None

# ==============================================================================
# AST Visitor to find tests and record their class context 
# ==============================================================================
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
        
# ==============================================================================
# Phase 1: Core code for data extraction 
# ==============================================================================
class TaskIdGenerator:
    def __init__(self):
        self.current_id = 0
    def next(self) -> int:
        task_id = self.current_id; self.current_id += 1; return task_id

class AssertionTransformer(ast.NodeTransformer):
    def __init__(self, target_lineno: int, target_col_offset: int):
        self.target_lineno, self.target_col_offset = target_lineno, target_col_offset
        self.transformed, self.ground_truth = False, None
    def visit_Assert(self, node: ast.Assert) -> ast.Assert:
        if not (node.lineno == self.target_lineno and node.col_offset == self.target_col_offset and
                isinstance(node.test, ast.Compare) and node.test.ops and isinstance(node.test.ops[0], ast.Eq)):
            return node
        original_node = node.test.comparators[0]
        self.ground_truth = astunparse.unparse(original_node).strip()
        self.transformed = True
        try: node.test.comparators[0] = ast.Constant(value='???', kind=None)
        except AttributeError: node.test.comparators[0] = ast.Name(id='???', ctx=ast.Load())
        return node

def process_rewritten_test_file(file_path: str, repo_root: str, task_id_gen: TaskIdGenerator):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: source_code = f.read()
        tree = ast.parse(source_code)
    except (SyntaxError, UnicodeDecodeError, PermissionError) as e:
        print(f"Warning: Failed to parse {file_path}. Reason: {e}. Skipped.", file=sys.stderr); return []

    imports_list = [astunparse.unparse(node).strip() for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    all_data_entries, reponame = [], os.path.basename(repo_root)
    original_testname = os.path.basename(file_path)
    normalized_testname = original_testname.replace('_agent_rewrite.py', '.py')
    original_relpath = os.path.relpath(file_path, repo_root)
    normalized_relpath = original_relpath.replace(original_testname, normalized_testname)
    
    quality_filter = create_quality_filter() if create_quality_filter else None

    finder = TestFinder()
    finder.visit(tree)

    for test_info in finder.found_tests:
        node, classname = test_info["node"], test_info["classname"]
        
        candidate_asserts = [item["assert_node"] for item in quality_filter.filter_test_function(node)] if quality_filter else [sn for sn in ast.walk(node) if isinstance(sn, ast.Assert) and isinstance(sn.test, ast.Compare) and sn.test.ops and isinstance(sn.test.ops[0], ast.Eq)]
        
        for assert_to_mask in candidate_asserts:
            func_copy = copy.deepcopy(node)
            transformer = AssertionTransformer(assert_to_mask.lineno, assert_to_mask.col_offset)
            masked_function_node = transformer.visit(func_copy)
            if transformer.transformed and transformer.ground_truth is not None:
                task_id_str = f"{reponame}_{task_id_gen.next()}"
                quality_analysis = quality_filter.analyze_assertion_complexity(assert_to_mask) if quality_filter else None
                
                data_entry = {
                    "task_id": task_id_str, "reponame": reponame,
                    "testpath": normalized_relpath.replace('\\', '/'),
                    "testname": normalized_testname,
                    "classname": classname,  # [NEW] Add classname field
                    "funcname": node.name,
                    "imports": imports_list, "code": astunparse.unparse(node).strip(),
                    "masked_code": astunparse.unparse(masked_function_node).strip(),
                    "ground_truth": transformer.ground_truth
                }
                if quality_analysis: data_entry["quality_analysis"] = quality_analysis
                all_data_entries.append(data_entry)
    return all_data_entries

def run_data_collection_stage(repos_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Extracted data will be temporarily saved to: {os.path.abspath(output_dir)}")
    repo_names = [d for d in os.listdir(repos_dir) if os.path.isdir(os.path.join(repos_dir, d))]
    if not repo_names:
        print(f"Error: No repository subdirectories found in '{repos_dir}'.", file=sys.stderr); return False
    for reponame in repo_names:
        repo_path = os.path.join(repos_dir, reponame)
        output_file = os.path.join(output_dir, f"{reponame}.jsonl")
        task_id_gen = TaskIdGenerator()
        print(f"\n--- Processing repository: {reponame} ---")
        all_processed_data = []
        for root, _, files in os.walk(repo_path):
            for file in files:
                if file.endswith("_agent_rewrite.py"):
                    file_path = os.path.join(root, file)
                    processed_data = process_rewritten_test_file(file_path, repo_path, task_id_gen)
                    all_processed_data.extend(processed_data)
        if not all_processed_data:
            print(f"No valid assertions found in {reponame}."); continue
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in all_processed_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"✅ Success! Found {len(all_processed_data)} assertions. Data saved to: {output_file}")
    return True

def normalize_func_name(name: str) -> str:
    return name.split('[')[0].split('::')[-1]
def get_difficulty(dependencies: list) -> str:
    if not dependencies: return 'easy'
    max_hops = max(dep.get('hops', 0) for dep in dependencies)
    return 'easy' if max_hops <= 2 else 'hard'
def load_difficulty_map(report_file_path: str) -> Union[Dict[str, str], None]:
    if not os.path.exists(report_file_path): return None
    difficulty_map = {}
    with open(report_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                norm_name = normalize_func_name(data['test_function'])
                if norm_name not in difficulty_map:
                    difficulty_map[norm_name] = get_difficulty(data.get('dependencies', []))
            except (json.JSONDecodeError, KeyError): continue
    return difficulty_map
def run_analysis_stage(collection_dir: str, reports_dir: str, final_output_dir: str):
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Final filtered data will be saved to: {os.path.abspath(final_output_dir)}")
    total_stats = {'lines_kept': 0, 'lines_discarded': 0, 'func_counts': defaultdict(int), 'total_unique_funcs': 0}
    data_files = [f for f in os.listdir(collection_dir) if f.endswith('.jsonl')]
    if not data_files:
        print(f"Warning: No .jsonl files found in temporary directory '{collection_dir}' for analysis.", file=sys.stderr); return
    for filename in data_files:
        reponame = filename.replace('.jsonl', '')
        print(f"\n--- Analyzing repository: {reponame} ---")
        report_file = os.path.join(reports_dir, reponame, 'report_functions.jsonl')
        difficulty_map = load_difficulty_map(report_file)
        if difficulty_map is None:
            print(f"  Warning: Report file '{report_file}' not found. Skipping analysis for this repo.\n"); continue
        input_path, output_path = os.path.join(collection_dir, filename), os.path.join(final_output_dir, filename)
        lines_kept, lines_discarded = 0, 0
        repo_func_stats, counted_funcs_in_repo = defaultdict(int), set()
        with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
            for line in infile:
                try:
                    assert_data = json.loads(line)
                    funcname = assert_data.get('funcname')
                    if funcname and funcname in difficulty_map:
                        outfile.write(line); lines_kept += 1
                        if funcname not in counted_funcs_in_repo:
                            difficulty = difficulty_map[funcname]
                            repo_func_stats[difficulty] += 1
                            counted_funcs_in_repo.add(funcname)
                    else: lines_discarded += 1
                except (json.JSONDecodeError, KeyError): lines_discarded += 1
        total_lines = lines_kept + lines_discarded
        print(f"  [Filter Stats] Processed {total_lines} lines:")
        if total_lines > 0:
            print(f"    - Kept: {lines_kept} lines ({lines_kept/total_lines:.1%})")
            print(f"    - Discarded: {lines_discarded} lines ({lines_discarded/total_lines:.1%})")
        num_unique_funcs = len(counted_funcs_in_repo)
        print(f"  [Difficulty Stats] Analyzed {num_unique_funcs} unique test functions in kept data:")
        if num_unique_funcs > 0:
            for difficulty in ['easy', 'hard']:
                count = repo_func_stats[difficulty]
                print(f"    - {difficulty.capitalize():<7}: {count} ({count/num_unique_funcs:.1%})")
        total_stats['lines_kept'] += lines_kept; total_stats['lines_discarded'] += lines_discarded
        total_stats['total_unique_funcs'] += num_unique_funcs
        for key, value in repo_func_stats.items(): total_stats['func_counts'][key] += value
    print("\n" + "=" * 60 + "\n--- Overall Analysis Report ---")
    grand_total_lines = total_stats['lines_kept'] + total_stats['lines_discarded']
    print(f"\n[Overall Filter Stats] Total processed {grand_total_lines} lines across all repos:")
    if grand_total_lines > 0:
        print(f"  - Total Kept: {total_stats['lines_kept']} lines ({total_stats['lines_kept']/grand_total_lines:.1%})")
        print(f"  - Total Discarded: {total_stats['lines_discarded']} lines ({total_stats['lines_discarded']/grand_total_lines:.1%})")
    total_funcs = total_stats['total_unique_funcs']
    print(f"\n[Overall Difficulty Stats] All kept data contains {total_funcs} unique test functions:")
    if total_funcs > 0:
        for difficulty in ['easy', 'hard']:
            count = total_stats['func_counts'][difficulty]
            print(f"  - {difficulty.capitalize():<7}: {count} ({count/total_funcs:.1%})")
    print("=" * 60)

def main():
    current_dir = os.getcwd()
    upper_dir = os.path.dirname(current_dir)
    rewrites_dir = os.path.join(upper_dir, 'output_rewrites')
    reports_dir = os.path.join(upper_dir, 'output_results')
    collection_temp_dir = os.path.join(upper_dir, 'data_collection_from_rewrites')
    final_filtered_dir = os.path.join(upper_dir, 'data_collection_from_rewrites_filtered')
    if not os.path.isdir(rewrites_dir) or not os.path.isdir(reports_dir):
        print(f"Error: Please ensure 'output_rewrites' and 'output_results' directories exist in the current working directory.", file=sys.stderr); return
    print("="*25 + " Phase 1: Extract Data " + "="*25)
    success = run_data_collection_stage(rewrites_dir, collection_temp_dir)
    if not success:
        print("Phase 1 failed to execute. Pipeline aborted.", file=sys.stderr); return
    print("\n" + "="*20 + " Phase 2: Analyze, Filter, and Categorize " + "="*20)
    run_analysis_stage(collection_temp_dir, reports_dir, final_filtered_dir)
    print("\n--- All tasks completed ---")

if __name__ == '__main__':
    try: import astunparse
    except ImportError:
        print("Error: 'astunparse' library is not installed.\nPlease install it using: pip install astunparse", file=sys.stderr); sys.exit(1)
    main()