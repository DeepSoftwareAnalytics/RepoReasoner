import json
import os
from pathlib import Path
import time
import datetime
import random
import re
import argparse
from tqdm import tqdm
from openai import OpenAI, APIError
import tiktoken
from prompt import PROMPT_TEMPLATE_CallChain_Prediction

# ==============================================================================
# Experiment Configuration
# ==============================================================================
class Config:
    # --- Dynamic API configuration ---
    API_BASE_URL = None
    API_KEY = None
    MODEL_NAME = None
    MODEL_CONTEXT_WINDOW = 20480
    
    # --- Static directory configuration ---
    DATA_DIR_BASE = Path("../Data_RepoReasoner")
    CALL_CHAIN_REPORTS_DIR = Path("../output_results")
    REPO_BASE_DIR = Path("../python_repos")

    EXPERIMENTS_OUTPUT_DIR = None

    # --- Experiment parameters ---
    CALL_CHAIN_RATIOS = [1.0]  # 100% positive files only
    MAX_COMPLETION_TOKENS = 1024
    MAX_ITEMS_PER_REPO = 200
    NUM_SAMPLES_PER_TASK = 5
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REQUEST_DELAY_SECONDS = 1

    # --- Tokenizer ---
    TOKENIZER_NAME = "o200k_base"
    
    # --- Derived constants ---
    MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
    SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 100

    # --- File scanning exclusions ---
    EXCLUDED_DIR_NAMES = {'.venv', '.git', 'docs', 'examples', '__pycache__', 'tests', 'test'}
    EXCLUDED_FILE_NAMES = {'__init__.py'}

    PROMPT_TEMPLATE = PROMPT_TEMPLATE_CallChain_Prediction

# ==============================================================================
# Helper Functions
# ==============================================================================
def count_tokens_custom(text, tok):
    return len(tok.encode(text))

def load_jsonl(file_path):
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: pass
    return data

def get_all_py_files(repo_root):
    py_files = []
    for root, dirs, files in os.walk(repo_root):
        dirs[:] = [d for d in dirs if d not in Config.EXCLUDED_DIR_NAMES and not d.startswith('.')]
        for file in files:
            if file.endswith('.py') and file not in Config.EXCLUDED_FILE_NAMES:
                full_path = Path(root) / file
                relative_path = full_path.relative_to(repo_root)
                py_files.append(str(relative_path))
    return py_files

def get_files_content_budgeted(files_to_include, reponame, token_budget, tokenizer):
    content_parts, included_files_list = [], []
    accumulated_tokens = 0
    random.shuffle(files_to_include)

    for file_path_str in files_to_include:
        if accumulated_tokens >= token_budget: break
        
        file_path = Config.REPO_BASE_DIR / reponame / file_path_str
        header = f"### file: {file_path_str}\n```python\n"
        footer = "\n```\n"
        header_tokens = count_tokens_custom(header, tokenizer)
        footer_tokens = count_tokens_custom(footer, tokenizer)

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content = f.read()
        except Exception: continue

        available_tokens = token_budget - accumulated_tokens
        if header_tokens + footer_tokens >= available_tokens: continue

        content_budget = available_tokens - header_tokens - footer_tokens
        if content_budget <= 0: continue

        token_ids = tokenizer.encode(file_content)
        if len(token_ids) > content_budget:
            token_ids = token_ids[:content_budget]
        
        final_content = tokenizer.decode(token_ids)
        full_part = header + final_content + footer
        part_tokens = count_tokens_custom(full_part, tokenizer)

        if accumulated_tokens + part_tokens <= token_budget:
            content_parts.append(full_part)
            accumulated_tokens += part_tokens
            included_files_list.append(file_path_str)
    
    return "".join(content_parts), included_files_list

# ==============================================================================
# Core Experiment Logic
# ==============================================================================
def process_repository(reponame, client, tokenizer):
    call_chain_report_file = Config.CALL_CHAIN_REPORTS_DIR / reponame / "report_files.jsonl"
    if not call_chain_report_file.exists(): return
    call_chain_lookup = {item['test_file']: [dep['file'] for dep in item['dependencies']] for item in load_jsonl(call_chain_report_file)}

    original_data_file = Config.DATA_DIR_BASE / "original" / f"{reponame}.jsonl"
    if not original_data_file.exists(): return
    
    seen_test_paths, tasks = set(), []
    for item in load_jsonl(original_data_file):
        test_path = item.get('testpath')
        if test_path and test_path in call_chain_lookup and test_path not in seen_test_paths:
            seen_test_paths.add(test_path)
            tasks.append({'task_id': item['task_id'], 'test_file': test_path})
            if len(tasks) >= Config.MAX_ITEMS_PER_REPO: break
    
    if not tasks: return

    all_repo_py_files = get_all_py_files(Config.REPO_BASE_DIR / reponame)

    for ratio in Config.CALL_CHAIN_RATIOS:
        # 100% positive files only
        ratio_str = "100_percent_positive"
        print(f"    -> Starting experiment: {ratio_str}")

        preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / "predictions" / ratio_str
        inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / "interactions" / ratio_str
        preds_dir.mkdir(parents=True, exist_ok=True)
        inters_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = preds_dir / f"{reponame}.jsonl"
        interactions_output_file = inters_dir / f"{reponame}.jsonl"

        with open(predictions_output_file, 'w', encoding='utf-8') as pred_f, \
             open(interactions_output_file, 'w', encoding='utf-8') as intr_f:

            tasks_to_process = []
            for task in tasks:
                positive_pool = [f for f in call_chain_lookup[task['test_file']] if f != task['test_file']]
                for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
                    tasks_to_process.append({"task_info": task, "positive_pool": positive_pool, "sample_idx": sample_idx})
            
            task_iterator = tqdm(tasks_to_process, desc=f"      Tasks", leave=False, position=1)
            for task_item in task_iterator:
                test_file_path = task_item['task_info']['test_file']
                
                try:
                    with open(Config.REPO_BASE_DIR / reponame / test_file_path, 'r', encoding='utf-8') as f: test_code = f.read()
                except Exception: continue

                static_prompt_tokens = count_tokens_custom(Config.PROMPT_TEMPLATE['system_message'], tokenizer) + count_tokens_custom(Config.PROMPT_TEMPLATE['user_prompt_template'].format(test_file_path="", test_code=test_code, files_context=""), tokenizer)
                files_total_budget = Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - static_prompt_tokens
                
                # 100% positive files only
                positive_files_budget = files_total_budget
                
                pos_content, included_pos = get_files_content_budgeted(task_item['positive_pool'], reponame, positive_files_budget, tokenizer)
                
                files_context_str = pos_content
                
                user_prompt = Config.PROMPT_TEMPLATE['user_prompt_template'].format(test_file_path=test_file_path, test_code=test_code, files_context=files_context_str)
                messages = [{"role": "system", "content": Config.PROMPT_TEMPLATE['system_message']}, {"role": "user", "content": user_prompt}]
                
                response_text, finish_reason, usage, error_details = "ERROR: INIT", "error", None, None
                try:
                    response = client.chat.completions.create(model=Config.MODEL_NAME, messages=messages, temperature=Config.TEMPERATURE, top_p=Config.TOP_P, max_tokens=Config.MAX_COMPLETION_TOKENS)
                    response_text, finish_reason = response.choices[0].message.content, response.choices[0].finish_reason
                    usage = response.usage.model_dump() if response.usage else None
                except APIError as e:
                    response_text, error_details = f"ERROR: APIError - {e.message}", {"type": type(e).__name__, "message": str(e), "status_code": e.status_code}
                except Exception as e:
                    response_text, error_details = f"ERROR: SDK/Network Error - {type(e).__name__}", {"type": type(e).__name__, "message": str(e)}

                task_info = task_item['task_info']
                intr_f.write(json.dumps({"task_id": task_info['task_id'], "test_file": task_info['test_file'], "sample_id": task_item['sample_idx'], "ratio": ratio, "messages": messages, "files_in_prompt": included_pos, "ground_truth_chain": task_item['positive_pool'], "error": error_details}, ensure_ascii=False) + "\n")
                pred_f.write(json.dumps({"task_id": task_info['task_id'], "test_file": task_info['test_file'], "sample_id": task_item['sample_idx'], "ratio": ratio, "response": response_text.strip(), "finish_reason": finish_reason, "usage": usage}, ensure_ascii=False) + "\n")
                time.sleep(Config.REQUEST_DELAY_SECONDS)

# ==============================================================================
# Main Function (Coordinator)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run call chain prediction experiment (API version).")
    
    parser.add_argument("--api_base_url", type=str, default='https://xxx/v1/', help="Base URL of the OpenAI-compatible API.")
    parser.add_argument("--api_key", type=str, default='sk-xxx', help="API key.")
    parser.add_argument("--model_name", type=str, default='qwen3-235b-a22b-instruct', help="Name of the model to use.")
    # gemini-2.5-flash-nothinking gpt-4.1-mini qwen3-235b-a22b-instruct
    parser.add_argument("--context_window", type=int, default=20480, help="Context window size of the model.")

    parser.add_argument("--ratios", nargs='+', type=float, default=[1.0], help="Token budget ratios for call chain files.")
    args = parser.parse_args()

    Config.API_BASE_URL = args.api_base_url
    Config.API_KEY = args.api_key
    Config.MODEL_NAME = args.model_name

    Config.MODEL_CONTEXT_WINDOW = args.context_window
    Config.CALL_CHAIN_RATIOS = args.ratios  
    
    model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.MODEL_NAME)
    Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_callchain_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}_oracle")
    Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING

    print("="*80)
    print("Experiment configuration initialized")
    print(f"  - API Base URL: {Config.API_BASE_URL}")
    print(f"  - Model Name: {Config.MODEL_NAME}")
    print(f"  - Context Window: {Config.MODEL_CONTEXT_WINDOW}")
    print(f"  - Call Chain Ratios: {Config.CALL_CHAIN_RATIOS} (100% positive files only)")
    print(f"  - Output Directory: {Config.EXPERIMENTS_OUTPUT_DIR}")
    print("="*80)

    random.seed(42)

    print("\n--- Initializing API client and Tokenizer ---")
    try:
        client = OpenAI(api_key=Config.API_KEY, base_url=Config.API_BASE_URL)
        tokenizer = tiktoken.get_encoding(Config.TOKENIZER_NAME)
        print("--- Initialization successful ---\n")
    except Exception as e:
        print(f"Error: Initialization failed: {e}"); exit(1)

    repo_names = sorted([p.stem for p in (Config.CALL_CHAIN_REPORTS_DIR).iterdir() if p.is_dir()])
    repo_iterator = tqdm(repo_names, desc="Repos", position=0)
    for reponame in repo_iterator:
        repo_iterator.set_postfix_str(f"Repo: {reponame}")
        if not (Config.REPO_BASE_DIR / reponame).exists(): continue
        process_repository(reponame, client, tokenizer)
            
    print("\n--- All experiments completed ---")

if __name__ == "__main__":
    main()