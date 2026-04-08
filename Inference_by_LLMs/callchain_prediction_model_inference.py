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
from prompt import PROMPT_TEMPLATE_CallChain_Prediction

# ==============================================================================
# Experiment configuration (partially set via command-line arguments)
# ==============================================================================
class Config:
    # --- Dynamic configuration ---
    LOCAL_MODEL_PATH = None
    TOKENIZER_PATH = None
    MODEL_CONTEXT_WINDOW = 20480
    EXPERIMENTS_OUTPUT_DIR = None
    CALL_CHAIN_RATIOS = [0.25, 0.5]  # Proportion of token budget allocated to call chain files

    # --- Core directory configuration ---
    DATA_DIR_BASE = Path("../Data_RepoReasoner")
    CALL_CHAIN_REPORTS_DIR = Path("../output_results")
    REPO_BASE_DIR = Path("../python_repos")

    # --- Experiment parameters ---
    MAX_COMPLETION_TOKENS = 1024
    MAX_ITEMS_PER_REPO = 200
    NUM_SAMPLES_PER_TASK = 5
    BATCH_SIZE = 4
    TEMPERATURE = 0.7
    TOP_P = 0.9

    # --- Derived constants ---
    MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = None
    SAFETY_MARGIN_TOKENS_PROMPT_BUILDING = 100

    # --- File scanning exclusions ---
    EXCLUDED_DIR_NAMES = {'.venv', '.git', 'docs', 'examples', '__pycache__', 'tests', 'test'}
    EXCLUDED_FILE_NAMES = {'__init__.py'}

    # ==============================================================================
    # Prompt template
    # ==============================================================================
    PROMPT_TEMPLATE = PROMPT_TEMPLATE_CallChain_Prediction


# ==============================================================================
# Helper functions
# ==============================================================================
def count_tokens_custom(text, tok):
    return len(tok.encode(text))


def load_jsonl(file_path):
    data = []
    if not file_path.exists():
        return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return data


def get_all_py_files(repo_root):
    """Get all valid Python files under a repository."""
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
    """Read and possibly truncate content of a group of files according to token budget, return concatenated string and list of included files."""
    content_parts, included_files_list = [], []
    accumulated_tokens = 0

    random.shuffle(files_to_include)  # Randomly select files to include

    for file_path_str in files_to_include:
        if accumulated_tokens >= token_budget:
            break

        file_path = Config.REPO_BASE_DIR / reponame / file_path_str
        header = f"### file: {file_path_str}\n```python\n"
        footer = "\n```\n"
        header_tokens = count_tokens_custom(header, tokenizer)
        footer_tokens = count_tokens_custom(footer, tokenizer)

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
        except Exception:
            continue

        available_tokens = token_budget - accumulated_tokens
        if header_tokens + footer_tokens >= available_tokens:
            continue

        content_budget = available_tokens - header_tokens - footer_tokens
        if content_budget <= 0:
            continue

        token_ids = tokenizer.encode(file_content)
        if len(token_ids) > content_budget:
            token_ids = token_ids[:content_budget]

        final_content = tokenizer.decode(token_ids, skip_special_tokens=True)
        full_part = header + final_content + footer
        part_tokens = count_tokens_custom(full_part, tokenizer)

        if accumulated_tokens + part_tokens <= token_budget:
            content_parts.append(full_part)
            accumulated_tokens += part_tokens
            included_files_list.append(file_path_str)

    return "".join(content_parts), included_files_list


# ==============================================================================
# Core experiment logic
# ==============================================================================
def process_repository(reponame, model, tokenizer):
    # --- 1. Data loading and preprocessing ---
    # Load ground truth call chains for this repository
    call_chain_report_file = Config.CALL_CHAIN_REPORTS_DIR / reponame / "report_files.jsonl"
    if not call_chain_report_file.exists():
        print(f"Warning: Call chain report 'report_files.jsonl' not found for repo {reponame}. Skipping.")
        return
    call_chain_lookup = {item['test_file']: [dep['file'] for dep in item['dependencies']] for item in load_jsonl(call_chain_report_file)}

    # Identify unique test files from data_collection_align as tasks
    original_data_file = Config.DATA_DIR_BASE / "original" / f"{reponame}.jsonl"
    if not original_data_file.exists():
        return

    seen_test_paths = set()
    tasks = []
    for item in load_jsonl(original_data_file):
        test_path = item.get('testpath')
        if test_path and test_path in call_chain_lookup and test_path not in seen_test_paths:
            seen_test_paths.add(test_path)
            tasks.append({'task_id': item['task_id'], 'test_file': test_path})
            if len(tasks) >= Config.MAX_ITEMS_PER_REPO:
                break

    if not tasks:
        print(f"No valid testable tasks found in repository {reponame}.")
        return

    # Get all possible code files within the repository
    all_repo_py_files = get_all_py_files(Config.REPO_BASE_DIR / reponame)

    # --- 2. Loop over different ratio experiments ---
    for ratio in Config.CALL_CHAIN_RATIOS:
        ratio_str = f"{int(ratio*100)}_percent_positive"
        print(f"    -> Starting experiment: {ratio_str}")

        # Set output directories
        preds_dir = Config.EXPERIMENTS_OUTPUT_DIR / "predictions" / ratio_str
        inters_dir = Config.EXPERIMENTS_OUTPUT_DIR / "interactions" / ratio_str
        preds_dir.mkdir(parents=True, exist_ok=True)
        inters_dir.mkdir(parents=True, exist_ok=True)
        predictions_output_file = preds_dir / f"{reponame}.jsonl"
        interactions_output_file = inters_dir / f"{reponame}.jsonl"

        with open(predictions_output_file, 'w', encoding='utf-8') as pred_f, \
             open(interactions_output_file, 'w', encoding='utf-8') as intr_f:

            # --- 3. Prepare batched tasks ---
            tasks_to_process = []
            for task in tasks:
                test_file = task['test_file']
                ground_truth_chain = call_chain_lookup[test_file]

                # Split into positive (call chain files) and negative (unrelated files)
                positive_pool = [f for f in ground_truth_chain if f != test_file]
                negative_pool = [f for f in all_repo_py_files if f not in ground_truth_chain]

                for sample_idx in range(Config.NUM_SAMPLES_PER_TASK):
                    tasks_to_process.append({
                        "task_info": task,
                        "positive_pool": positive_pool,
                        "negative_pool": negative_pool,
                        "sample_idx": sample_idx
                    })

            # --- 4. Batch processing ---
            task_iterator = tqdm(range(0, len(tasks_to_process), Config.BATCH_SIZE), desc=f"      Batches", leave=False, position=1)
            for i in task_iterator:
                batch = tasks_to_process[i:i + Config.BATCH_SIZE]
                prompts_batch, messages_batch, metadata_batch = [], [], []

                for task_item in batch:
                    test_file_path = task_item['task_info']['test_file']

                    try:
                        with open(Config.REPO_BASE_DIR / reponame / test_file_path, 'r', encoding='utf-8') as f:
                            test_code = f.read()
                    except Exception as e:
                        continue

                    # Calculate token budget
                    static_prompt_tokens = count_tokens_custom(Config.PROMPT_TEMPLATE['system_message'], tokenizer) + \
                                         count_tokens_custom(Config.PROMPT_TEMPLATE['user_prompt_template'].format(test_file_path="", test_code=test_code, files_context=""), tokenizer)

                    files_total_budget = Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION - static_prompt_tokens
                    positive_files_budget = int(files_total_budget * ratio)
                    negative_files_budget = files_total_budget - positive_files_budget

                    # Fill positive and negative file contents based on budget
                    pos_content, included_pos = get_files_content_budgeted(task_item['positive_pool'], reponame, positive_files_budget, tokenizer)
                    neg_content, included_neg = get_files_content_budgeted(task_item['negative_pool'], reponame, negative_files_budget, tokenizer)

                    # Combine and shuffle context parts
                    context_parts = [pos_content, neg_content]
                    random.shuffle(context_parts)
                    files_context_str = "".join(context_parts)

                    # Construct final prompt
                    user_prompt = Config.PROMPT_TEMPLATE['user_prompt_template'].format(
                        test_file_path=test_file_path,
                        test_code=test_code,
                        files_context=files_context_str
                    )
                    messages = [
                        {"role": "system", "content": Config.PROMPT_TEMPLATE['system_message']},
                        {"role": "user", "content": user_prompt}
                    ]

                    text_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts_batch.append(text_prompt)
                    messages_batch.append(messages)
                    task_item['included_files'] = included_pos + included_neg
                    metadata_batch.append(task_item)

                if not prompts_batch:
                    continue

                # --- 5. Batch inference and saving ---
                try:
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    model_inputs = tokenizer(
                        prompts_batch,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION
                    ).to(model.device)
                    generated_ids = model.generate(
                        **model_inputs,
                        max_new_tokens=Config.MAX_COMPLETION_TOKENS,
                        do_sample=True,
                        temperature=Config.TEMPERATURE,
                        top_p=Config.TOP_P,
                        pad_token_id=tokenizer.eos_token_id
                    )
                    response_texts = tokenizer.batch_decode(
                        generated_ids[:, model_inputs.input_ids.shape[1]:],
                        skip_special_tokens=True
                    )
                    finish_reason, error_details = "stop", None
                except Exception as e:
                    response_texts = [f"ERROR: {type(e).__name__}"] * len(batch)
                    finish_reason, error_details = "error", {"type": type(e).__name__, "message": str(e)}

                for idx, response_text in enumerate(response_texts):
                    task_info = metadata_batch[idx]
                    intr_f.write(json.dumps({
                        "task_id": task_info['task_info']['task_id'],
                        "test_file": task_info['task_info']['test_file'],
                        "sample_id": task_info['sample_idx'],
                        "ratio": ratio,
                        "messages": messages_batch[idx],
                        "files_in_prompt": task_info['included_files'],
                        "ground_truth_chain": task_info['positive_pool'],
                        "error": error_details
                    }, ensure_ascii=False) + "\n")
                    pred_f.write(json.dumps({
                        "task_id": task_info['task_info']['task_id'],
                        "test_file": task_info['task_info']['test_file'],
                        "sample_id": task_info['sample_idx'],
                        "ratio": ratio,
                        "response": response_text.strip(),
                        "finish_reason": finish_reason
                    }, ensure_ascii=False) + "\n")


# ==============================================================================
# Main function (orchestrator)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Run Call Chain Prediction experiment.")
    parser.add_argument("--model_path", type=str, required=True, help="Local path to Hugging Face model.")
    parser.add_argument("--context_window", type=int, default=20480, help="Model's context window size.")
    parser.add_argument("--ratios", nargs='+', type=float, default=[0.25, 0.5], help="Call chain file token budget ratios to run, e.g., 0.25 0.5.")
    args = parser.parse_args()

    # --- Dynamically update configuration ---
    Config.LOCAL_MODEL_PATH = Path(args.model_path)
    Config.TOKENIZER_PATH = Config.LOCAL_MODEL_PATH
    Config.MODEL_CONTEXT_WINDOW = args.context_window
    Config.CALL_CHAIN_RATIOS = args.ratios
    model_name_slug = Config.LOCAL_MODEL_PATH.name
    Config.EXPERIMENTS_OUTPUT_DIR = Path(f"experiments_callchain_{model_name_slug}_{Config.MODEL_CONTEXT_WINDOW}")
    Config.MAX_TOKENS_FOR_PROMPT_CONSTRUCTION = Config.MODEL_CONTEXT_WINDOW - Config.MAX_COMPLETION_TOKENS - Config.SAFETY_MARGIN_TOKENS_PROMPT_BUILDING

    print("="*80)
    print("Experiment configuration initialized")
    print(f"  - Model path: {Config.LOCAL_MODEL_PATH}")
    print(f"  - Context window: {Config.MODEL_CONTEXT_WINDOW}")
    print(f"  - Call chain ratios: {Config.CALL_CHAIN_RATIOS}")
    print(f"  - Output directory: {Config.EXPERIMENTS_OUTPUT_DIR}")
    print("="*80)

    random.seed(42)

    # --- Load model and tokenizer ---
    print("\n--- Loading model ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            Config.LOCAL_MODEL_PATH,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        print("--- Model loaded successfully ---\n")
    except Exception as e:
        print(f"Error: Failed to initialize model: {e}")
        exit(1)

    # --- Run experiments ---
    repo_names = sorted([p.stem for p in (Config.CALL_CHAIN_REPORTS_DIR).iterdir() if p.is_dir()])
    repo_iterator = tqdm(repo_names, desc="Repos", position=0)
    for reponame in repo_iterator:
        repo_iterator.set_postfix_str(f"Repo: {reponame}")
        if not (Config.REPO_BASE_DIR / reponame).exists():
            print(f"Warning: Repository source code '{Config.REPO_BASE_DIR / reponame}' not found, skipping.")
            continue
        process_repository(reponame, model, tokenizer)

    print("\n--- All experiments completed ---")


if __name__ == "__main__":
    main()