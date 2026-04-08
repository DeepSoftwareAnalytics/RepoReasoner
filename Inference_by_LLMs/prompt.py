PROMPT_TEMPLATE_Output_Prediction = {
"system_message": """Your task is to infer the correct values for the "???" placeholder based on provided Python code.
# Rules
1.  Analyze the provided code and related files to determine the value.
2.  The value MUST be a determined value or a valid Python expression.
3.  **Do not** output any explanations, reasoning, introductions, or any other text outside of the specified output format. Your output must be clean.
# Output format

```output
answer_of_placeholder
```

# Example
Here is an example of a perfect interaction.

## Related files are as follow:
### file: config.py
```python
DEFAULT_TIMEOUT = 50
```

### file: get_timeout.py
```python
from config import DEFAULT_TIMEOUT
def get_timeout():
    return DEFAULT_TIMEOUT
```

## Test function is as follows:
### test
```python
def test_timeout():
    from get_timeout import get_timeout
    assert get_timeout() == "???"
```

### Your Correct Output:
```output
50
```
""",
    "user_prompt_template": """## Related files are as follows:
{related_files_content}

## Test function is as follows:
```python
{masked_code_content}
```

Based on all the provided context and the test function, infer the correct value or expression for the "???" placeholder. Output it directly in the required format. Do not output extra information.
"""
}


PROMPT_TEMPLATE_CallChain_Prediction = {
"system_message": """You are an expert Python static analysis assistant. Your task is to analyze a test file and a set of provided code files to identify ONLY the files that are necessary for the execution of the test's call chain.

# Rules
1.  Carefully trace all imports and function calls starting from the test file to identify its dependencies among the provided files.
2.  The test file itself is the starting point but MUST NOT be included in the final output list.
3.  Your output MUST be ONLY a single, clean JSON list of file path strings.
4.  **Do not** output any explanations, reasoning, analysis steps, or any other text. Your response must contain only the final JSON in a markdown block.

# Output Format
```json
["path/to/dependency1.py", "path/to/dependency2.py"]
```

# Example
Here is an example of a perfect interaction.

## Candidate files are as follows:
### file: my_app/calculator.py
```python
def add(a, b):
    return a + b
```
### file: my_app/utils.py
```python
import os
def check_file_exists(path):
    return os.path.exists(path)
```

## The test file is as follows:
### file: tests/test_calculator.py
```python
from my_app.calculator import add
def test_addition():
    assert add(2, 3) == 5
```

## Your Correct Output:
```json
["my_app/calculator.py"]
```
""",

"user_prompt_template": """## The following is a pool of candidate files from the repository.
{files_context}

## The test file is as follows:
### file: {test_file_path}
```python
{test_code}
```
Based on all the provided code, identify the file paths that are part of the test's call chain. Output ONLY the final JSON list as instructed in the rules.
"""
}