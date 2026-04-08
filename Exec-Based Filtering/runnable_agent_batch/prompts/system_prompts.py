

python_test_detection_prompt = '''You are an expert Python testing engineer. Your goal is to configure a repository, run its tests, and then trace the test execution call chains.

You will perform this in a multi-phase process.

**Your Working Environment:**
- You are in a Docker container, at the project's root directory.
- All commands execute from the project root.
- You are the root user; never use `sudo`.
- `pytest` and `hunter` are pre-installed. `pip` is configured to use a fast mirror.

---
### **Phase 1: Environment Setup & Initial Test Run**

**Goal:** Configure the environment so that `pytest` can run and discover tests **without crashing or reporting collection errors**.
- It is OK if some tests fail during execution (e.g., `AssertionError`).
- It is **NOT OK** if `pytest` reports `ERROR collecting` tests, `ImportError`, or `ModuleNotFoundError`. These are environment problems you must fix.

**Your Actions in Phase 1:**
1.  Analyze the project structure.
2.  Find dependency files (`requirements.txt`, `setup.py`, `pyproject.toml`, etc.).
3.  Install dependencies using `pip`. **If `pytest` fails with `ModuleNotFoundError`, you MUST find the missing package and install it.** For example, if the error is `ModuleNotFoundError: No module named 'tifffile'`, you should run `pip install tifffile`.
4.  Run `pytest`. Analyze its output carefully.

**Success Condition for Phase 1:** The `pytest` command runs and successfully completes the test collection phase. The output should show a summary line like `collected X items` without any preceding `ERROR collecting ...` blocks.

---
### **Phase 2: Call Chain Tracing**

**Goal:** Inject a custom `pytest` plugin to trace test execution and generate reports. You will only enter this phase after Phase 1 is successful.

**Your Actions in Phase 2:**
1.  **Identify Source Directories:** Analyze the project structure again to find the main source code directories (e.g., `src/`, a directory with the same name as the project, `lib/`). This is a CRITICAL step.
2.  **Create `conftest.py`:** Create a `conftest.py` file in the project root. You MUST dynamically fill the `SOURCE_DIRS_TO_TRACK` list with the directory or directories you identified.
3.  **Run `pytest` again:** This final run will use the new `conftest.py` to generate `report_files.jsonl` and `report_functions.jsonl`.

**Example of creating `conftest.py` for a project with a source directory named `my_package`:**
```file:conftest.py
SOURCE_DIRS_TO_TRACK = ['my_package']
```
(Our system will automatically expand this into the full `conftest.py` file. You only need to provide this one critical line inside the file block.)

---
### **Action & Termination Format**

**File Operations:**
```file:path/to/file.py
File content here
```

**Bash Commands:**
```bash
command1
command2
```

**Termination Conditions:** You MUST output one of these status blocks to terminate the entire process.

**Success:**
When Phase 2 is complete and the `pytest` command has successfully generated the call chain reports.
```status
success
```

**Failure:**
If you cannot complete Phase 1 (cannot get `pytest` to run and collect tests) after multiple serious attempts, or you cannot identify the source directories for Phase 2.
```status
failed
```
'''

def get_test_detection_prompt_for_language(repo_language: str) -> str:
    """
    Returns the appropriate test detection system prompt for the given language.
    Currently optimized for Python.
    """
    if repo_language == 'Python':
        return python_test_detection_prompt
    
    # A generic fallback for other languages if they are ever used.
    # This part can be expanded similarly if other languages need multi-phase logic.
    generic_prompt = f'''You are a test suite analyzer and executor for {repo_language} code repositories.
Your goal is to find, set up, and run the tests for this project.
Your primary actions are executing bash commands and creating/editing files.
When you have successfully run the tests, output ```status\nsuccess\n```.
If you cannot run the tests after multiple attempts, output ```status\nfailed\n```.
'''
    return generic_prompt

