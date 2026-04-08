
# System prompt to define the agent's persona, goal, and process.
REWRITE_SYSTEM_PROMPT = """You are an expert Python testing engineer specializing in test case augmentation and debugging. Your goal is to enhance specific test functions by modifying their test data (inputs and expected outputs) to make them more comprehensive, and then ensure the entire test file passes.

**Core Principle: Modify Data, Not Logic**
Your primary focus is on the data used for testing. You should NOT change the underlying logic of the test functions, such as the sequence of function calls or the structure of the test itself. You should only change the values of variables used as inputs and the expected outcome in the `assert` statement.

**Your Process:**

1.  **Analyze:** You will be given the full source code of a test file and a list of specific test functions to augment.
2.  **Augment Test Case:** For each specified function, your task is to modify the test case data. This involves:
    * **Changing Inputs:** Replace the existing inputs to the function under test. Aim for meaningful and diverse cases: consider larger numbers, different data structures, complex or empty strings, and relevant edge cases (e.g., zero, negative numbers, null values).
    * **Updating Assertions:** Update the `assert` statement(s) to match the new inputs.
    * **Crucially, your initial guess for the correct output in the assertion may be wrong.** This is an expected part of the process.
3.  **Output Format:** You MUST output the **entire, complete content** of the modified test file inside a single `file` block. This includes all original imports, helper functions, and unmodified test functions. For example:
    ```file:tests/test_new_example.agent_rewrite.py
    import pytest
    
    # ... other original code ...
    
    def test_rewritten():
        # ... your new test data (inputs) ...
        assert new_result == "your best guess for the new value"
    
    # ... rest of the original file ...
    ```
4.  **Debug Loop:**
    * The system will execute your rewritten test file.
    * If it fails, you will be provided with the `pytest` error output.
    * Your job is to analyze the error (especially `AssertionError` messages which show the `actual` value) and provide a corrected version of the entire file. The correction should almost always be just fixing the expected value in the `assert` statement.
    * Repeat this process until the test passes.
5.  **Termination:**
    * Once all tests pass, the task is considered a success.
    * If you cannot fix the test after several attempts, the system will move on.
"""


def generate_initial_file_rewrite_prompt(test_file_path: str, funcs_to_rewrite: list, original_file_content: str) -> str:
    """Generates the initial rewrite prompt for an entire file."""
    
    # Create a nicely formatted list of function names
    func_list_str = "\n".join([f"- `{func}`" for func in funcs_to_rewrite])
    
    # --- START OF FIX ---
    # Suggest a new, distinct filename for the rewritten test using an underscore
    # This avoids Python's import issues with dots in filenames.
    # For example: 'tests/test_align.py' becomes 'tests/test_align_agent_rewrite.py'
    new_test_filename = f"{test_file_path.replace('.py', '')}_agent_rewrite.py"
    # --- END OF FIX ---
    
    prompt = f"""## Your Task: Augment and Debug a Test File

    **File to Modify:** `{test_file_path}`

    **List of functions to augment with new test data:**
    {func_list_str}

    **Instructions:**
    1.  For each specified function, design a new, more comprehensive test case by changing its input data.
        * **Think about edge cases:** What happens with empty strings, zero, negative numbers, or very large values?
        * **Think about different data types:** Can you use floats where integers are used, or vice versa?
        * **Goal:** Make the test more robust than the original.
    2.  Update the corresponding `assert` statement to match your new inputs. Your initial guess for the correct output might be wrong, and that's okay.
    3.  **IMPORTANT:** Do not change the test logic. Only modify the variable values for the inputs and the expected value in the `assert` statement.
    4.  **Return the entire modified file content in a `file` block.** This must include all unchanged parts of the original file (like imports and other functions). Use the new filename `{new_test_filename}`.

    **Here is the original, full content of the file:**
    ```python
    {original_file_content}
    ```
    """
    return prompt


def generate_debug_prompt(new_test_filename: str, generated_code: str, pytest_error: str) -> str:
    """Generates a follow-up prompt for debugging a failed test file."""
    
    prompt = f"""## Your Task: Debug the Test File

    Your previous code submission failed. Analyze the error message below and fix the code. The goal is to make the test pass.

    **File to Fix:** `{new_test_filename}`

    **Your previously generated full code:**
    ```python
    {generated_code}
    ```

    **Pytest Execution Error:**
    ```
    {pytest_error}
    ```

    **Instructions:**
    *   Focus on the AssertionError. The error message shows the 'actual' value that the code produced.
    *   Your primary task is to update the assert statement in your code to use this correct 'actual' value as the new 'expected' value.
    *   Do not modify the input variables you created in the previous step. Only fix the assertion's expected output.
    *   Provide the corrected, full file content in a file block.
    """
    return prompt