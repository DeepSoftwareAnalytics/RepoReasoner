# -*- coding: utf-8 -*-


import os
import re
from pathlib import Path
from typing import List, Tuple

def disable_call_chain_tracking_in_conftest(conftest_path: Path) -> Tuple[bool, str]:
    if not conftest_path.exists():
        return False, ""
    
    try:
        content = conftest_path.read_text(encoding='utf-8')
        
        if 'hunter.trace' not in content and 'FileTracer' not in content:
            return False, content  
        
        modified_content = _disable_tracking_functions(content)

        conftest_path.write_text(modified_content, encoding='utf-8')
        
        return True, content
        
    except Exception as e:
        print(f"modify conftest.py failed: {e}")
        return False, ""

def restore_conftest_content(conftest_path: Path, original_content: str) -> bool:
    try:
        conftest_path.write_text(original_content, encoding='utf-8')
        return True
    except Exception as e:
        print(f"restore conftest.py failed: {e}")
        return False

def _disable_tracking_functions(content: str) -> str:
    lines = content.split('\n')
    modified_lines = []
    i = 0
    
    while i < len(lines):
        line = lines[i]
        
        if 'def pytest_runtest_protocol(' in line:
            modified_lines.append(line)
            modified_lines.append('    # Call chain tracking disabled during rewrite phase')
            modified_lines.append('    yield')
            modified_lines.append('    return')

            i += 1
            indent_level = len(line) - len(line.lstrip())
            while i < len(lines):
                current_line = lines[i]
                if current_line.strip() and len(current_line) - len(current_line.lstrip()) <= indent_level:
                    break
                i += 1
            continue
        
        elif 'def pytest_sessionfinish(' in line:
            modified_lines.append(line)
            modified_lines.append('    # Call chain tracking disabled during rewrite phase')
            modified_lines.append('    return')
            
            i += 1
            indent_level = len(line) - len(line.lstrip())
            while i < len(lines):
                current_line = lines[i]
                if current_line.strip() and len(current_line) - len(current_line.lstrip()) <= indent_level:
                    break
                i += 1
            continue
        
        else:
            modified_lines.append(line)
            i += 1
    
    return '\n'.join(modified_lines)

def find_and_disable_conftest_files(repo_path: Path) -> List[Tuple[Path, str]]:
    modified_files = []
    
    for conftest_file in repo_path.rglob("conftest.py"):
        success, original_content = disable_call_chain_tracking_in_conftest(conftest_file)
        if success:
            modified_files.append((conftest_file, original_content))
            print(f"disable conftest.py: {conftest_file}")
    
    return modified_files

def restore_all_conftest_files(modified_files: List[Tuple[Path, str]]):
    for conftest_file, original_content in modified_files:
        if restore_conftest_content(conftest_file, original_content):
            print(f"restore yet: {conftest_file}")
        else:
            print(f"restore failed: {conftest_file}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) != 2:
        print("python conftest_disable_tracking.py <repo_path>")
        sys.exit(1)
    
    repo_path = Path(sys.argv[1])
    if not repo_path.exists():
        print(f"path is not exist: {repo_path}")
        sys.exit(1)
    
    print(f"In {repo_path}, disable contest.py...")
    modified_files = find_and_disable_conftest_files(repo_path)
    
    if modified_files:
        print(f"modify {len(modified_files)} conftest.py")
        input("enter...")
        restore_all_conftest_files(modified_files)
    else:
        print("conftest.py not find")
