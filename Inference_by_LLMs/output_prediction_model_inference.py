import json
import os
from pathlib import Path
import time
import datetime
import random
import torch
import re
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from prompt import PROMPT_TEMPLATE_Output_Prediction
# ==============================================================================
# Experiment configuration
# ==============================================================================
class Config:
    # These values will be overwritten by command-line arguments
    LOCAL_MODEL_PATH = None
    TOKENIZER_PATH = None
    MODEL_CONTEXT_WINDOW = None 
    EXPERIMENTS_OUTPUT_DIR = None

    # --- Core directory configuration ---
    BASE_DATA_DIR = Path("../Data_RepoReasoner")
    BM25_RANK_DIR = Path("../output_with_bm25_rank")
    CALL_CHAIN_DATA_DIR = Path("../output_results")
    REPO_BASE_DIR = Path("../python_repos")
    
    # --- Experimental parameters ---
    MAX_COMPLETION_TOKENS = 512
    MAX_ITEMS_PER_REPO = 200
    NUM_SAMPLES_PER_TASK = 5
    BATCH_SIZE = 8  # Batch size, can be adjusted based on VRAM
    TEMPERATURE = 0.7
    TOP_P = 0.9

    # --- File selection parameters for different experiment types ---
    # [Retrieval]
    MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL = 20
    # [Confusion]
    NUM_CONFUSION_FILES_TO_SAMPLE = 20
    TOP_N_TO_EXCLUDE_FOR_CONFUSION = 20
    # [Oracle]
    MAX_ORACLE_FILES_TO_CONSIDER = 20 # Assume Oracle also uses an upper limit

    # --- Derived constants (will be computed in main function) ---
    MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
    SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 50

    # ==============================================================================
    # Prompt templates
    # ==============================================================================
    PROMPT_TEMPLATE_PARTS = PROMPT_TEMPLATE_Output_Prediction

# ==============================================================================
# Helper functions
# ==============================================================================
def count_tokens_custom(text, tok):
    """Count the number of tokens in the text using the specified tokenizer."""
    return len(tok.encode(text))

def load_jsonl(file_path):
    """Load a JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for ln, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decoding error at line {ln+1} in {file_path}: {e}. Line content: '{line.strip()}'")
    return data

def get_related_files_content_budgeted_custom(reponame_arg, related_files_rank_list, current_tokenizer, token_budget_for_related_files):
    """Get and format the content of related files within a given token budget."""
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
        except Exception as e: file_content_str = f"# Read error for file {file_path}: {e}"
            
        available_tokens_for_this_file = token_budget_for_related_files - accumulated_tokens
        if file_header_tokens + file_footer_tokens >= available_tokens_for_this_file: break
        
        current_file_str_parts = [file_header_str]
        current_file_tokens = file_header_tokens
        content_tokens_budget = available_tokens_for_this_file - file_header_tokens - file_footer_tokens
        
        if content_tokens_budget > 0:
            content_token_ids = current_tokenizer.encode(file_content_str)
            if len(content_token_ids) > content_tokens_budget:
                truncated_content_token_ids = content_token_ids[:content_tokens_budget]
                truncated_content_str = current_tokenizer.decode(truncated_content_token_ids, skip_special_tokens=True)
                current_file_str_parts.append(truncated_content_str)
                current_file_tokens += count_tokens_custom(truncated_content_str, current_tokenizer)
            else:
                current_file_str_parts.append(file_content_str)
                current_file_tokens += len(content_token_ids)

        current_file_str_parts.append(file_footer_str)
        current_file_tokens += file_footer_tokens

        if accumulated_tokens + current_file_tokens <= token_budget_for_related_files:
            content_parts.append("".join(current_file_str_parts))
            accumulated_tokens += current_file_tokens
        else: break
            
    if not content_parts:
        final_str = "# No related file content could be included within the token limit.\n"
        return final_str, count_tokens_custom(final_str, current_tokenizer)
    return "".join(content_parts), accumulated_tokens

def normalize_funcname(name: str) -> str:
    """Aggressively normalize function name by removing parameterized parts and class paths."""
    name = re.sub(r'\[.*\]$', '', name)
    if '::' in name:
        name = name.split('::')[-1]
    return name

# ==============================================================================
# Core experimental logic
# ==============================================================================
def process_repository(reponame, dataset_dir, condition_name, model, tokenizer):
    """Process all experiment types for a single repository."""
    dataset_file = dataset_dir / f"{reponame}.jsonl"
    bm25_file = Config.BM25_RANK_DIR / f"{reponame}.jsonl"
    call_chain_file = Config.CALL_CHAIN_DATA_DIR / reponame / "report_functions.jsonl"

    if not dataset_file.exists():
        print(f"Error: Dataset file {dataset_file} not found.")
        return

    test_data_list = load_jsonl(dataset_file)[:Config.MAX_ITEMS_PER_REPO]
    
    # Load data required for various retrieval strategies
    bm25_lookup = load_jsonl(bm25_file) if bm25_file.exists() else []
    bm25_lookup = {item['task_id']: item for item in bm25_lookup}

    call_chain_data = load_jsonl(call_chain_file) if call_chain_file.exists() else []
    call_chain_lookup = {}
    for item in call_chain_data:
        dirty_func_name = item.get('test_function')
        if not dirty_func_name or 'dependencies' not in item: continue
        normalized_name = normalize_funcname(dirty_func_name)
        sorted_deps = sorted(item['dependencies'], key=lambda x: x.get('hops', float('inf')))
        call_chain_lookup[normalized_name] = [dep['file'] for dep in sorted_deps]

    # Pre-calculate tokens for prompt template
    tokens_system_message = count_tokens_custom(Config.PROMPT_TEMPLATE_PARTS["system_message"], tokenizer)
    static_user_prompt_template = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(related_files_content="", masked_code_content="")
    tokens_static_user_prompt = count_tokens_custom(static_user_prompt_template, tokenizer)

    # Loop through all experiment types
    # for experiment_type in ["retrieval", "confusion", "oracle"]:
    for experiment_type in ["retrieval", "oracle"]:
        print(f"    -> Starting experiment: {experiment_type}")
        
        # Check if required files exist
        if experiment_type in ["retrieval", "confusion"] and not bm25_lookup:
            print(f"      Warning: BM25 ranking file {bm25_file} not found, skipping {experiment_type} experiment.")
            continue
        if experiment_type == "oracle" and not call_chain_lookup:
            print(f"      Warning: Call chain file {call_chain_file} not found, skipping {experiment_type} experiment.")
            continue

        # Set output file paths
        current_preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "predictions" / experiment_type
        current_inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / condition_name / "interactions" / experiment_type
        current_preds_dir.mkdir(parents=True, exist_ok=True)
        current_inters_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = current_preds_dir / f"{reponame}.jsonl"
        interactions_output_file = current_inters_dir / f"{reponame}.jsonl"

        with open(predictions_output_file, 'w', encoding='utf-8') as pred_f, \
             open(interactions_output_file, 'w', encoding='utf-8') as intr_f:
            
            tasks_to_process = []
            for test_case in test_data_list:
                task_id = test_case['task_id']
                related_files_for_experiment = []

                # Select related files based on experiment type
                if experiment_type == "retrieval":
                    if task_id in bm25_lookup:
                        all_ranked_files = bm25_lookup[task_id].get('related_files_rank', [])
                        related_files_for_experiment = all_ranked_files[:Config.MAX_RELATED_FILES_TO_CONSIDER_RETRIEVAL]
                        # [Fix] Perform reproducible shuffling to eliminate positional bias
                        random.shuffle(related_files_for_experiment)
                
                elif experiment_type == "confusion":
                    if task_id in bm25_lookup:
                        all_ranked_files = bm25_lookup[task_id].get('related_files_rank', [])
                        confusion_pool = all_ranked_files[Config.TOP_N_TO_EXCLUDE_FOR_CONFUSION:]
                        if len(confusion_pool) >= Config.NUM_CONFUSION_FILES_TO_SAMPLE:
                           related_files_for_experiment = random.sample(confusion_pool, Config.NUM_CONFUSION_FILES_TO_SAMPLE)
                        else:
                           related_files_for_experiment = confusion_pool

                elif experiment_type == "oracle":
                    test_name = test_case.get('funcname')
                    normalized_test_name = normalize_funcname(test_name) if test_name else ""
                    if normalized_test_name in call_chain_lookup:
                        related_files_from_call_chain = call_chain_lookup[normalized_test_name]
                        related_files_for_experiment = related_files_from_call_chain[:Config.MAX_ORACLE_FILES_TO_CONSIDER]
                
                if not related_files_for_experiment and experiment_type != 'confusion':
                     pass

                for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
                    tasks_to_process.append({
                        "test_case": test_case,
                        "related_files": related_files_for_experiment,
                        "sample_idx": sample_idx
                    })
            
            # Batch processing
            task_iterator = tqdm(range(0, len(tasks_to_process), Config.BATCH_SIZE), desc=f"      Batches", leave=False, position=1, bar_format='{l_bar}{bar:10}{r_bar}')
            for i in task_iterator:
                batch = tasks_to_process[i:i + Config.BATCH_SIZE]
                
                prompts_batch = []
                messages_batch = []
                metadata_batch = []

                for task_item in batch:
                    test_case = task_item["test_case"]
                    related_files = task_item["related_files"]
                    
                    masked_code = test_case['masked_code']
                    tokens_test_code = count_tokens_custom(masked_code, tokenizer)
                    token_budget_for_dynamic_content = (Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - tokens_system_message - tokens_static_user_prompt - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING)
                    token_budget_for_related_files = token_budget_for_dynamic_content - tokens_test_code
                    
                    related_files_str, _ = get_related_files_content_budgeted_custom(reponame, related_files, tokenizer, token_budget_for_related_files) if token_budget_for_related_files > 0 and related_files else ("# No related file context.\n", 0)
                    
                    user_prompt = Config.PROMPT_TEMPLATE_PARTS["user_prompt_template"].format(
                        related_files_content=related_files_str,
                        masked_code_content=masked_code
                    )
                    messages = [{"role": "system", "content": Config.PROMPT_TEMPLATE_PARTS["system_message"]}, {"role": "user", "content": user_prompt}]
                    
                    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts_batch.append(text_prompt)
                    messages_batch.append(messages)
                    metadata_batch.append(task_item)

                # Batch inference
                response_texts, finish_reason, error_details = [], "stop", None
                try:
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token

                    model_inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True, max_length=Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION).to(model.device)
                    
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=Config.MAX_COMPLETION_TOKENS,
                        do_sample=True,
                        temperature=Config.TEMPERATURE,
                        top_p=Config.TOP_P,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    
                    response_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
                    response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

                except torch.cuda.OutOfMemoryError as e:
                    print(f"      Fatal Error: CUDA out of memory! Please try reducing BATCH_SIZE. {e}"); exit(1)
                except Exception as e:
                    print(f"      Unexpected error during local inference: {e}")
                    response_texts = [f"ERROR: LocalInference - {type(e).__name__}"] * len(batch)
                    finish_reason = "local_sdk_error"
                    error_details = {"type": type(e).__name__, "message": str(e)}

                # Save batch results
                for idx, response_text in enumerate(response_texts):
                    task_info = metadata_batch[idx]
                    task_id = task_info["test_case"]["task_id"]
                    sample_idx = task_info["sample_idx"]
                    
                    intr_f.write(json.dumps({
                        "task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx,
                        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        "messages_sent_to_model_template": messages_batch[idx], 
                        "model_used": str(Config.LOCAL_MODEL_PATH),
                        "related_files_provided_to_prompt": task_info["related_files"], 
                        "api_error_details": error_details
                    }, ensure_ascii=False) + "\n")
                    pred_f.write(json.dumps({
                        "task_id": task_id, "experiment_type": experiment_type, "sample_id": sample_idx, 
                        "response": response_text.strip(), 
                        "finish_reason": finish_reason, "usage": None
                    }, ensure_ascii=False) + "\n")

# ==============================================================================
# Main function (coordinator)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run code completion experiments supporting multiple retrieval strategies.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the local Hugging Face model to load.")
    parser.add_argument("--context_window", type=int, required=True, help="Context window size of the model.")
    args = parser.parse_args()

    # --- Dynamically update configuration ---
    Config.LOCAL_MODEL_PATH = Path(args.model_path)
    Config.TOKENIZER_PATH = Config.LOCAL_MODEL_PATH
    Config.MODEL_CONTEXT_WINDOW = args.context_window
    model_name_slug = Config.LOCAL_MODEL_PATH.name
    Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_output_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}")
    Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - 100
    
    print("="*80)
    print(f"Experiment configuration initialized")
    print(f"  - Model path: {Config.LOCAL_MODEL_PATH}")
    print(f"  - Context window: {Config.MODEL_CONTEXT_WINDOW}")
    print(f"  - Output directory: {Config.EXPERIMENTS_OUTPUT_DIR}")
    print("="*80)

    # [Core] Set random seed for reproducibility
    random.seed(42)

    # --- Load model and tokenizer ---
    print("\n--- Starting to load local model (native multi-GPU mode) ---")
    if not Config.LOCAL_MODEL_PATH.exists() or not (Config.LOCAL_MODEL_PATH / "config.json").exists():
        print(f"Fatal error: Model path '{Config.LOCAL_MODEL_PATH}' does not exist or is invalid.")
        exit(1)
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            Config.LOCAL_MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        print("--- Model loaded and distributed across all available GPUs successfully ---\n")
    except Exception as e:
        print(f"Error: Failed to initialize local model: {e}"); exit(1)

    # --- Run experiments ---
    data_conditions = {"original": Config.BASE_DATA_DIR / "original", "rewrite": Config.BASE_DATA_DIR / "rewrite"}
    for condition_name, dataset_dir in data_conditions.items():
        print("\n" + "="*80)
        print(f"Starting data condition: '{condition_name.upper()}'")
        print("="*80)
        
        if not dataset_dir.exists():
            print(f"Error: Data directory '{dataset_dir}' does not exist. Skipping.")
            continue
        
        repo_names = sorted([p.stem for p in dataset_dir.glob("*.jsonl")])
        if not repo_names:
            print(f"No .jsonl files found in {dataset_dir}.")
            continue
        
        repo_iterator = tqdm(repo_names, desc="Repos", position=0, bar_format='{l_bar}{bar:20}{r_bar}')
        for reponame in repo_iterator:
            repo_iterator.set_postfix_str(f"Repo: {reponame}")
            process_repository(reponame, dataset_dir, condition_name, model, tokenizer)
            
    print("\n--- All experiment conditions completed ---")

if __name__ == "__main__":
    main()