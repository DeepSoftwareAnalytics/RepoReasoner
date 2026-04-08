import os
from pathlib import Path
from runnable_agent_batch.utils import get_project_structure, truncate_content

# The full, correct content of the conftest.py tracer.
# This template is used by the run.py script to ensure the correct file is written,
# even if the model only provides a partial response.
CONFTEST_PY_CONTENT = """
import pytest
import hunter
from collections import defaultdict
import json
import os
from pathlib import Path

# NOTE: The agent MUST replace this with the correct source directory/directories.
SOURCE_DIRS_TO_TRACK = ['<placeholder_source_dir>'] 

# --- The rest of the code is boilerplate ---
test_call_chains = defaultdict(dict)
try:
    # When conftest.py is at the project root, its parent is not the project root.
    # So, we define the project root as the directory containing conftest.py.
    PROJECT_ROOT = str(Path(__file__).resolve().parent)
except (NameError, FileNotFoundError):
    PROJECT_ROOT = os.getcwd()

# Ensure SOURCE_PATHS are absolute for reliable prefix matching
SOURCE_PATHS = [os.path.realpath(os.path.join(PROJECT_ROOT, d)) for d in SOURCE_DIRS_TO_TRACK]
if not SOURCE_PATHS or '<placeholder_source_dir>' in SOURCE_DIRS_TO_TRACK:
    print(
        "WARNING: Call chain tracking is enabled, but SOURCE_DIRS_TO_TRACK is not set correctly in conftest.py. "
        "No source files will be tracked."
    )

class FileTracer:
    def __init__(self, test_nodeid):
        self.test_nodeid = test_nodeid
        self.source_file_stack = []

    def __call__(self, event):
        if event.kind not in ('call', 'return'):
            return
        
        # event.filename can be None for certain built-in calls
        module_realpath = getattr(event, 'filename', None)
        if not module_realpath:
            return

        is_source_file = any(module_realpath.startswith(p) for p in SOURCE_PATHS)
        if not is_source_file:
            return

        last_file_on_stack = self.source_file_stack[-1] if self.source_file_stack else None
        
        if event.kind == 'call':
            if module_realpath != last_file_on_stack:
                self.source_file_stack.append(module_realpath)
                dependencies = test_call_chains[self.test_nodeid]
                current_hops = len(self.source_file_stack)
                if module_realpath not in dependencies or current_hops < dependencies[module_realpath]:
                    dependencies[module_realpath] = current_hops
        elif event.kind == 'return':
            if module_realpath == last_file_on_stack:
                self.source_file_stack.pop()

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    # Check if call chain tracking is disabled
    if os.environ.get('DISABLE_CALL_CHAIN_TRACKING'):
        yield
        return
    
    # Create a new tracer instance for each test item
    with hunter.trace(action=FileTracer(item.nodeid)):
        yield

def pytest_sessionfinish(session):
    # Check if call chain tracking is disabled
    if os.environ.get('DISABLE_CALL_CHAIN_TRACKING'):
        return
    
    function_level_data = test_call_chains
    if not function_level_data:
        print("\\n--- Call Chain Report ---")
        print("No call chains were recorded. (Hint: Check SOURCE_DIRS_TO_TRACK in conftest.py)")
        return

    # --- Function-level Report ---
    func_report_filename = "report_functions.jsonl"
    print(f"\\n--- Writing function-level test call chains report to {func_report_filename} ---")
    try:
        with open(os.path.join(PROJECT_ROOT, func_report_filename), "w", encoding="utf-8") as f:
            for test_nodeid, dependencies in sorted(function_level_data.items()):
                dep_list = [{"file": os.path.relpath(p, PROJECT_ROOT), "hops": h} for p, h in dependencies.items()]
                dep_list.sort(key=lambda x: (x['hops'], x['file']))

                try:
                    test_file, test_func = test_nodeid.split("::", 1)
                except ValueError:
                    test_file, test_func = test_nodeid, "[module-level]"

                record = {"test_file": test_file, "test_function": test_func, "dependencies": dep_list}
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        print(f"Successfully wrote {len(function_level_data)} records to {func_report_filename}")
    except IOError as e:
        print(f"\\nError writing function-level report to file: {e}")

    # --- Aggregated File-level Report ---
    file_level_data = defaultdict(dict)
    for test_nodeid, dependencies in function_level_data.items():
        test_file_path = test_nodeid.split("::")[0]
        aggregated_deps = file_level_data[test_file_path]
        for dep_path, hops in dependencies.items():
            if dep_path not in aggregated_deps or hops < aggregated_deps[dep_path]:
                aggregated_deps[dep_path] = hops
                
    file_report_filename = "report_files.jsonl"
    print(f"\\n--- Writing aggregated file-level test call chains report to {file_report_filename} ---")
    try:
        with open(os.path.join(PROJECT_ROOT, file_report_filename), "w", encoding="utf-8") as f:
            for test_file, dependencies in sorted(file_level_data.items()):
                dep_list = [{"file": os.path.relpath(p, PROJECT_ROOT), "hops": h} for p, h in dependencies.items()]
                dep_list.sort(key=lambda x: (x['hops'], x['file']))
                record = {"test_file": test_file, "dependencies": dep_list}
                f.write(json.dumps(record, ensure_ascii=False) + "\\n")
        print(f"Successfully wrote {len(file_level_data)} records to {file_report_filename}")
    except IOError as e:
        print(f"\\nError writing file-level report to file: {e}")
    print("\\n--- End of Reports ---")
"""


def generate_initial_user_prompt(repo_path: Path, language: str, encoding=None) -> str:
    """Generate the initial user prompt with project structure for Phase 1."""
    project_structure = get_project_structure(repo_path)
    
    prompt = f'''## Project Information
Project Name: {repo_path.name}
Language: {language}

## Project Structure
```
{project_structure}
```

## Your Task (Phase 1: Environment Setup)
Your first task is to set up the environment and make `pytest` runnable.
1.  Find and install dependencies.
2.  Run `pytest`.

If `pytest` runs and collects tests (even with test failures), the next step will be to trace call chains. If you cannot make it run, declare failure.
'''
    return prompt


# file path: runnable_agent_batch/prompts/user_prompts.py

def generate_followup_user_prompt(
    repo_path: Path,
    language: str,
    phase: str,
    execution_results: str = None, 
    file_results: str = None,
    encoding=None
) -> str:
    """Generate a follow-up prompt that is aware of the current phase."""
    
    current_structure = get_project_structure(repo_path, encoding=encoding)
    
    prompt_parts = [
        f"## Current Project Structure",
        "```",
        current_structure,
        "```",
        ""
    ]
    
    if execution_results or file_results:
        prompt_parts.append("## Previous Operation Results:")
        if file_results:
            prompt_parts.append(f"### File Operations:\n{truncate_content(file_results, 1000, encoding)}")
        if execution_results:
            
            prompt_parts.append(f"### Command Execution:\n{truncate_content(execution_results, 6000, encoding)}")

    followup_content = "\n".join(prompt_parts)
    
    if phase == 'SETUP':
        task_prompt = """## Your Task (Phase 1: Environment Setup)
**Analyze the `pytest` output above.**
- Did `pytest` report `ModuleNotFoundError` or `ImportError`? If so, you have found a missing dependency. Your next action MUST be to install it using `pip`. For example, if you see `ModuleNotFoundError: No module named 'tifffile'`, respond with `pip install tifffile`.
- Did `pytest` report `ERROR collecting test`? This indicates a problem with the test files themselves or the environment setup. Continue to debug.
- You are still in Phase 1. Do NOT proceed to Phase 2 until these errors are resolved and `pytest` can collect tests without crashing.
- If you are truly stuck after several attempts, declare failure with ```status\nfailed\n```."""
    else:  # phase == 'TRACING'
        task_prompt = f"""## Your Task (Phase 2: Call Chain Tracing)
`pytest` is now executable without collection errors. Your task is to set up call chain tracing.
1.  **Identify the main source code directory/directories.** Based on the structure, this is likely `skimage`.
2.  **Create a `conftest.py` file** in the project root to trace this directory.
3.  **Run `pytest`** one last time to generate the reports.
4.  If the reports are generated, declare success with ```status\nsuccess\n```.

**REMINDER:** When creating `conftest.py`, you only need to provide the line defining `SOURCE_DIRS_TO_TRACK`. For example:
```file:conftest.py
SOURCE_DIRS_TO_TRACK = ['skimage']
```
"""

    return f"{followup_content}\n\n{task_prompt}"
