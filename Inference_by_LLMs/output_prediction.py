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
import httpx 
from prompt import PROMPT_TEMPLATE_Output_Prediction
# ==============================================================================
# Experimental Config
# ==============================================================================
class Config:
    # --- Dynamic API configuration (set via command-line arguments) ---
    API_BASE_URL = None
    API_KEY = None
    MODEL_NAME = None
    MODEL_CONTEXT_WINDOW = None
    
    # --- Static directory configuration ---
    BASE_DATA_DIR = Path("../Data_RepoReasoner")
    BM25_RANK_DIR = Path("../output_with_bm25_rank")
    CALL_CHAIN_DATA_DIR = Path("../output_results")
    REPO_BASE_DIR = Path("../python_repos")
    EXPERIMENTS_OUTPUT_DIR = None

    # --- Experiment parameters ---
    MAX_COMPLETION_TOKENS = 512
    MAX_ITEMS_PER_REPO = 200
    NUM_SAMPLES_PER_TASK = 5
    TEMPERATURE = 0.7
    TOP_P = 0.9
    REQUEST_DELAY_SECONDS = 1

    # --- Tokenizer ---
    TOKENIZER_NAME = "o200k_base"

    # --- File selection parameters ---
    MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL = 20
    NUM_CONFUSION_FILES_TO_SAMPLE = 20
    TOP_N_TO_EXCLUDE_FOR_CONFUSION = 20
    MAX_ORACLE_FILES_TO_CONSIDER = 20

    # --- Derived constants ---
    MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
    SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 50
    
    # Add network robustness parameters, does not affect experiment logic
    API_TIMEOUT_SECONDS = 60.0

    PROMPT_TEMPLATE_PARTS = PROMPT_TEMPLATE_Output_Prediction

# ==============================================================================
# Helper functions
# ==============================================================================
def count_tokens_custom(text, tok):
    return len(tok.encode(text))

def load_jsonl(file_path):
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data

def get_related_files_content_budgeted_custom(reponame_arg, related_files_rank_list, current_tokenizer, token_budget_for_related_files):
    content_parts, accumulated_tokens = [], 0
    if token_budget_for_related_files <= 0: return "# No token budget allocated for related files.\n", 0
    for rel_path_str in related_files_rank_list:
        if accumulated_tokens >= token_budget_for_related_files: break
        file_path = Config.REPO_BASE_DIR / reponame_arg / rel_path_str
        file_header_str = f"### file: {rel_path_str}\n```python\n"
        file_footer_str = "\n```\n"
        file_header_tokens = count_tokens_custom(file_header_str, current_tokenizer)
        file_footer_tokens = count_tokens_custom(file_footer_str, current_tokenizer)
        
        file_content_str = ""
        try:
            if file_path.exists() and file_path.is_file():
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f: file_content_str = f.read()
            else: file_content_str = f"# File {file_path} not found"
        except Exception as e: file_content_str = f"# Error reading file {file_path}: {e}"
            
        available_tokens_for_this_file = token_budget_for_related_files - accumulated_tokens
        if file_header_tokens + file_footer_tokens >= available_tokens_for_this_file: break
        
        current_file_str_parts = [file_header_str]
        content_tokens_budget = available_tokens_for_this_file - file_header_tokens - file_footer_tokens
        
        if content_tokens_budget > 0:
            content_token_ids = current_tokenizer.encode(file_content_str)
            if len(content_token_ids) > content_tokens_budget:
                truncated_content_token_ids = content_token_ids[:content_tokens_budget]
                truncated_content_str = current_tokenizer.decode(truncated_content_token_ids)
                current_file_str_parts.append(truncated_content_str)
            else:
                current_file_str_parts.append(file_content_str)
        current_file_str_parts.append(file_footer_str)
        
        full_part = "".join(current_file_str_parts)
        part_tokens = count_tokens_custom(full_part, current_tokenizer)
        
        if accumulated_tokens + part_tokens <= token_budget_for_related_files:
            content_parts.append(full_part)
            accumulated_tokens += part_tokens
        else: break
            
    if not content_parts:
        return "# No related file content can be included within the token limit.\n", 0
    return "".join(content_parts), accumulated_tokens

def normalize_funcname(name: str) -> str:
    name = re.sub(r'\[.*\]$', '', name)
    if '::' in name:
        name = name.split('::')[-1]
    return name

# ==============================================================================
# Core experiment logic
# ==============================================================================
def process_repository(reponame, dataset_dir, condition_name, client, tokenizer):
    dataset_file = dataset_dir / f"{reponame}.jsonl"
    bm25_file = Config.BM25_RANK_DIR / f"{reponame}.jsonl"
    call_chain_file = Config.CALL_CHAIN_DATA_DIR / reponame / "report_functions.jsonl"
    if not dataset_file.exists(): return

    test_data_list = load_jsonl(dataset_file)[:Config.MAX_ITEMS_PER_REPO]
    bm25_lookup = {item['task_id']: item for item in load_jsonl(bm25_file)}
    call_chain_data = load_jsonl(call_chain_file)
    call_chain_lookup = {}
    for item in call_chain_data:
        dirty_func_name = item.get('test_function')
        if not dirty_func_name: continue
        normalized_name = normalize_funcname(dirty_func_name)
        sorted_deps = sorted(item.get('dependencies', []), key=lambda x: x.get('hops', float('inf')))
        call_chain_lookup[normalized_name] = [dep['file'] for dep in sorted_deps]

    tokens_system_message = count_tokens_custom(Config.PROMPT_TEMPLATE_PARTS["system_message"], tokenizer)
    static_user_prompt_template = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content="", masked_code_content="")
    tokens_static_user_prompt = count_tokens_custom(static_user_prompt_template, tokenizer)

    # for experiment_type in ["retrieval", "confusion", "oracle"]:
    for experiment_type in ["retrieval", "oracle"]:
        print(f"    -> Starting experiment: {experiment_type}")
        current_preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "predictions" / experiment_type
        current_inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "interactions" / experiment_type
        current_preds_dir.mkdir(parents=True, exist_ok=True)
        current_inters_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = current_preds_dir / f"{reponame}.jsonl"
        interactions_output_file = current_inters_dir / f"{reponame}.jsonl"

        # Resume from breakpoint: Load completed tasks
        completed_tasks = set()
        if predictions_output_file.exists():
            for item in load_jsonl(predictions_output_file):
                task_key = (item.get('task_id'), item.get('sample_id'))
                completed_tasks.add(task_key)
        
        # Only print message if there are completed tasks
        if completed_tasks:
             print(f"      Found {len(completed_tasks)} completed tasks, will skip them.")

        # Open files in append mode ('a')
        with open(predictions_output_file, 'a', encoding='utf-8') as pred_f, \
             open(interactions_output_file, 'a', encoding='utf-8') as intr_f:
            
            tasks_to_process = []
            for test_case in test_data_list:
                task_id = test_case['task_id']
                related_files_for_experiment = []

                if experiment_type == "retrieval":
                    if task_id in bm25_lookup:
                        files = bm25_lookup[task_id].get('related_files_rank', [])
                        related_files_for_experiment = files[:Config.MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL]
                        random.shuffle(related_files_for_experiment)
                elif experiment_type == "confusion":
                    if task_id in bm25_lookup:
                        files = bm25_lookup[task_id].get('related_files_rank', [])
                        pool = files[Config.TOP_N_TO_EXCLUDE_FOR_CONFUSION:]
                        related_files_for_experiment = random.sample(pool, min(len(pool), Config.NUM_CONFUSION_FILES_TO_SAMPLE))
                elif experiment_type == "oracle":
                    func_name = test_case.get('funcname')
                    if func_name and normalize_funcname(func_name) in call_chain_lookup:
                        files = call_chain_lookup[normalize_funcname(func_name)]
                        related_files_for_experiment = files[:Config.MAX_ORACLE_FILES_TO_CONSIDER]
                
                for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
                    tasks_to_process.append({"test_case": test_case, "related_files": related_files_for_experiment, "sample_idx": sample_idx})
            
            task_iterator = tqdm(tasks_to_process, desc=f"      Tasks", leave=False, position=1)
            for task_item in task_iterator:
                test_case, related_files, sample_idx = task_item["test_case"], task_item["related_files"], task_item["sample_idx"]
                task_id = test_case["task_id"]

                # Resume from breakpoint: Check and skip if already done
                task_key = (task_id, sample_idx)
                if task_key in completed_tasks:
                    continue

                masked_code = test_case['masked_code']
                tokens_test_code = count_tokens_custom(masked_code, tokenizer)
                token_budget_for_dynamic_content = (Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - tokens_system_message - tokens_static_user_prompt - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING)
                token_budget_for_related_files = token_budget_for_dynamic_content - tokens_test_code
                
                related_files_str, _ = get_related_files_content_budgeted_custom(reponame, related_files, tokenizer, token_budget_for_related_files) if token_budget_for_related_files > 0 else ("# No related file context.\n", 0)
                
                user_prompt = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content=related_files_str, masked_code_content=masked_code)
                messages = [{"role": "system", "content": Config.PROMPT_TEMPLATE_PARTS["system_message"]}, {"role": "user", "content": user_prompt}]
                
                response_text, finish_reason, usage, error_details = "ERROR: INIT", "error", None, None

                # More comprehensive error handling
                try:
                    response = client.chat.completions.create(
                        model=Config.MODEL_NAME, messages=messages, temperature=Config.TEMPERATURE,
                        top_p=Config.TOP_P, max_tokens=Config.MAX_COMPLETION_TOKENS
                    )
                    response_text = response.choices[0].message.content
                    finish_reason = response.choices[0].finish_reason
                    usage = response.usage.model_dump() if response.usage else None
                except APIError as e:
                    print(f"\n      Captured API error (Task: {task_id}): {type(e).__name__} - {e}")
                    response_text = f"ERROR: {type(e).__name__} - {str(e)}"
                    status_code = getattr(e, 'status_code', None) # Safely get status_code
                    error_details = {"type": type(e).__name__, "message": str(e), "status_code": status_code}
                except Exception as e:
                    print(f"\n      Captured unexpected error (Task: {task_id}): {type(e).__name__} - {e}")
                    response_text = f"ERROR: SDK/Network Error - {str(e)}"
                    error_details = {"type": type(e).__name__, "message": str(e)}

                intr_f.write(json.dumps({"task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "messages_sent_to_model": messages, "model_used": Config.MODEL_NAME, "related_files_provided_to_prompt": related_files, "api_error_details": error_details}, ensure_ascii=False) + "\n")
                pred_f.write(json.dumps({"task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, "response": response_text.strip(), "finish_reason": finish_reason, "usage": usage}, ensure_ascii=False) + "\n")
                time.sleep(Config.REQUEST_DELAY_SECONDS)

# ==============================================================================
# Main function (coordinator)
# ==============================================================================
def main():
    # -------------------------: Command-line argument parsing --------------------------
    parser = argparse.ArgumentParser(description="Code prediction experiment script (supports OpenAI-compatible API)")
    
    # 1. API core arguments (required)
    parser.add_argument("--api-key", required=True, help="API key for OpenAI-compatible service (e.g., sk-xxx)")
    parser.add_argument("--api-base-url", required=True, help="API base URL (e.g., https://xxx/v1/)")
    parser.add_argument("--model-name", required=True, help="Model name (e.g., gemini-2.5-flash-nothinking)")
    parser.add_argument("--model-context-window", required=True, type=int, help="Model context window size (e.g., 10240)")
    
    # 2. Optional arguments (default values can be omitted)
    parser.add_argument("--api-timeout", type=float, default=60.0, help="API request timeout in seconds, default 60")
    parser.add_argument("--request-delay", type=float, default=1.0, help="Delay between requests in seconds, default 1")
    parser.add_argument("--tokenizer-name", default="o200k_base", help="Tokenizer name, default o200k_base")
    
    # Parse arguments
    args = parser.parse_args()

    # API core configuration
    Config.API_BASE_URL = args.api_base_url
    Config.API_KEY = args.api_key
    Config.MODEL_NAME = args.model_name
    Config.MODEL_CONTEXT_WINDOW = args.model_context_window
    
    # Optional parameter configuration
    Config.API_TIMEOUT_SECONDS = args.api_timeout
    Config.REQUEST_DELAY_SECONDS = args.request_delay
    Config.TOKENIZER_NAME = args.tokenizer_name


    # CUSTOM_API_KEY = "sk-xxx"
    # CUSTOM_API_BASE_URL = "https:/xxx/v1/"
    # OPENAI_COMPATIBLE_MODEL_NAME = "gemini-2.5-flash-nothinking"
    
    # Config.API_BASE_URL = CUSTOM_API_BASE_URL 
    # Config.API_KEY = CUSTOM_API_KEY
    # Config.MODEL_NAME = OPENAI_COMPATIBLE_MODEL_NAME
    Config.MODEL_CONTEXT_WINDOW = 10240
    model_name_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', Config.MODEL_NAME)
    Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_output_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}")
    Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - 100
    


    print("="*80)
    print("Experiment configuration initialized")
    print(f"  - API Base URL: {Config.API_BASE_URL}")
    print(f"  - Model Name: {Config.MODEL_NAME}")
    print(f"  - Context window: {Config.MODEL_CONTEXT_WINDOW}")
    print(f"  - Output directory: {Config.EXPERIMENTS_OUTPUT_DIR}")
    print("="*80)

    random.seed(42)

    print("\n--- Initializing API client and Tokenizer ---")
    try:
        http_client = None
        
        # Set timeout at the OpenAI client level
        client = OpenAI(
            api_key=Config.API_KEY, 
            base_url=Config.API_BASE_URL, 
            http_client=http_client,
            timeout=Config.API_TIMEOUT_SECONDS # Set default timeout for all requests
        )
        tokenizer = tiktoken.get_encoding(Config.TOKENIZER_NAME)
        print("--- Initialization successful ---\n")
    except Exception as e:
        print(f"Error: Initialization failed: {e}"); exit(1)

    data_conditions = {"original": Config.BASE_DATA_DIR / "original", "rewrite": Config.BASE_DATA_DIR / "rewrite"}
    for condition_name, dataset_dir in data_conditions.items():
        print(f"\n{'='*80}\nStarting data condition: '{condition_name.upper()}'\n{'='*80}")
        if not dataset_dir.exists():
            print(f"Error: Data directory '{dataset_dir}' does not exist. Skipping.")
            continue
        
        repo_names = sorted([p.stem for p in dataset_dir.glob("*.jsonl")])
        repo_iterator = tqdm(repo_names, desc="Repos", position=0)
        for reponame in repo_iterator:
            repo_iterator.set_postfix_str(f"Repo: {reponame}")
            process_repository(reponame, dataset_dir, condition_name, client, tokenizer)
            
    print("\n--- All experiment conditions completed ---")

if __name__ == "__main__":
    main()