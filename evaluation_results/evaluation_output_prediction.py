import os
import json
import re
import math
import argparse
import ast
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    PREDICTIONS_DIR_BASE = None
    EVALUATION_RESULTS_DIR = None
    DATA_DIR_BASE = Path("../Data_RepoReasoner")
    ANSWER_PATTERN = re.compile(r"(?:```output\s*\n|^)(.*?)(?:\s*```|$)", re.DOTALL)
    MODEL_SPECIFIC_PATTERNS = {
        "deepseek-r1-distill-qwen-14b": re.compile(r"```output\s*\n(.*?)\s*```", re.DOTALL),
        "qwen-2.5-coder-14b": re.compile(r"```output\s*\n(.*?)\s*```", re.DOTALL),
        "qwen-2.5-14b": re.compile(r"```output\s*\n(.*?)\s*```", re.DOTALL),
    }
    K_VALUES_TO_CALCULATE = [1, 2, 3, 4, 5]
    K_VALUES_TO_DISPLAY = [1, 5]
    NUM_SAMPLES_PER_TASK = 5
    EXCLUDED_TASK_IDS = []

def load_jsonl(file_path):
    data = []
    if not file_path.exists(): return data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try: data.append(json.loads(line))
            except json.JSONDecodeError: print(f"⚠️ WARNING: Skipping malformed JSON line in {file_path}")
    return data

def robust_compare(prediction_str: str, ground_truth_str: str) -> bool:
    prediction_str = prediction_str.strip()
    ground_truth_str = ground_truth_str.strip()
    if prediction_str == ground_truth_str: return True
    try:
        pred_obj = ast.literal_eval(prediction_str)
        gt_obj = ast.literal_eval(ground_truth_str)
        if type(pred_obj) == type(gt_obj): return pred_obj == gt_obj
    except (ValueError, SyntaxError, MemoryError, TypeError): return False
    return False

def is_prediction_correct(prediction: str, ground_truth_list: list) -> bool:
    for gt_val in ground_truth_list:
        if robust_compare(prediction, str(gt_val)): return True
    return False
    
def combinations(n, k):
    if k < 0 or k > n: return 0
    return math.comb(n, k)

def calculate_pass_at_k(n, c, k):
    if n - c < k: return 1.0
    denominator = combinations(n, k)
    if denominator == 0: return 0.0
    return 1.0 - (combinations(n - c, k) / denominator)


def print_summary_table(all_results: list):
    if not all_results:
        print("No results were collected to generate a summary table.")
        return

    df = pd.DataFrame(all_results)
    
    if 'row_key' in df.columns:
        df = df.drop(columns=['row_key'])

    fixed_cols = ['Model', 'Condition']
    all_metric_cols = [col for col in df.columns if col not in fixed_cols]

    ordered_metric_cols = []
    data_sizes_order = ['10k', '30k']
    modes_order = ['retrieval', 'oracle', 'confusion']
    k_values_order = Config.K_VALUES_TO_DISPLAY

    for size in data_sizes_order:
        for mode in modes_order:
            for k in k_values_order:
                col_name = f"{size}_{mode}_pass@{k}"
                if col_name in all_metric_cols:
                    ordered_metric_cols.append(col_name)

    remaining_cols = sorted(list(set(all_metric_cols) - set(ordered_metric_cols)))
    
    ordered_cols = fixed_cols + ordered_metric_cols + remaining_cols
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ordered_cols]
    
    for col in (ordered_metric_cols + remaining_cols):
        if col in df.columns and df[col].dtype == 'float64':
            df[col] = df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) else '---')

    df.fillna('---', inplace=True)

    print("\n\n" + "="*40 + " FINAL SUMMARY " + "="*40)
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    print("="*95 + "\n")

def evaluate_experiment(exp_dir: Path, all_results_collector: list):
    print(f"\n{'='*25} EVALUATING EXPERIMENT: {exp_dir.name} {'='*25}")

    dir_name = exp_dir.name
    
    match = re.match(r'experiments_output_(.+)_(10240|30720)$', dir_name)
    if not match:
        print(f"⚠️ WARNING: Skipping directory '{dir_name}' as it doesn't match the expected pattern '..._model_name_contextlen'.")
        return
        
    base_model_name = match.group(1)
    context_len = match.group(2)
    data_size_key = "10k" if context_len == "10240" else "30k"
    
    print(f"  - Base Model: {base_model_name}")
    print(f"  - Context Config (Data Size Key): {data_size_key}")

    Config.PREDICTIONS_DIR_BASE = exp_dir
    Config.EVALUATION_RESULTS_DIR = exp_dir.parent / exp_dir.name.replace("experiments_output_", "evaluation_results_")
    Config.EVALUATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"  - Output reports will be saved to: {Config.EVALUATION_RESULTS_DIR}")

    current_answer_pattern = Config.ANSWER_PATTERN
    for model_key, specific_pattern in Config.MODEL_SPECIFIC_PATTERNS.items():
        if model_key in base_model_name:
            current_answer_pattern = specific_pattern; break

    for mode in ["retrieval", "confusion", "oracle"]:
        print(f"--- Evaluating Mode: {mode.upper()} for {data_size_key} ---")
        
        if not (Config.PREDICTIONS_DIR_BASE / "original" / "predictions" / mode).exists():
            print(f"Prediction directory for mode '{mode}' not found. Skipping.")
            continue

        output_files = {}
        for cond in ["original", "rewrite"]:
            output_dir = Config.EVALUATION_RESULTS_DIR / cond / mode
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file_path = output_dir / "evaluation.jsonl"
            output_files[cond] = open(output_file_path, 'w', encoding='utf-8')

        all_ground_truths = {"original": {}, "rewrite": {}}
        all_predictions = {"original": defaultdict(list), "rewrite": defaultdict(list)}
        repo_list = sorted([f.stem for f in (Config.PREDICTIONS_DIR_BASE / "original" / "predictions" / mode).glob('*.jsonl')])
        for reponame in repo_list:
            for cond in ["original", "rewrite"]:
                gt_file = Config.DATA_DIR_BASE / cond / f"{reponame}.jsonl"
                for item in load_jsonl(gt_file): all_ground_truths[cond][item['task_id']] = item
                pred_file = Config.PREDICTIONS_DIR_BASE / cond / "predictions" / mode / f"{reponame}.jsonl"
                for pred in load_jsonl(pred_file): all_predictions[cond][pred['task_id']].append(pred)

        common_task_ids = set(all_predictions["original"].keys()).intersection(set(all_predictions["rewrite"].keys()))
        
        if Config.EXCLUDED_TASK_IDS:
            excluded_set = set(Config.EXCLUDED_TASK_IDS)
            original_count = len(common_task_ids)
            common_task_ids = common_task_ids - excluded_set
            num_excluded = original_count - len(common_task_ids)
            if num_excluded > 0:
                print(f"  - INFO: Excluding {num_excluded} task_id(s) based on the global exclusion list.")

        if not common_task_ids: 
            print("  - INFO: No common task IDs left to evaluate after exclusion. Skipping mode.")
            for f in output_files.values(): f.close()
            continue

        final_summary = defaultdict(lambda: {"pass_k": defaultdict(float), "num_problems": 0})

        for task_id in sorted(list(common_task_ids)):
            for cond in ["original", "rewrite"]:
                gt_item = all_ground_truths[cond].get(task_id)
                if not gt_item: continue
                
                samples = all_predictions[cond][task_id]
                
                num_correct = 0
                extracted_predictions = []
                for s in samples:
                    response = s.get('response', '')
                    match = current_answer_pattern.search(response)
                    extracted_pred = match.group(1).strip() if match else ""
                    extracted_predictions.append(extracted_pred)
                    if match and is_prediction_correct(extracted_pred, gt_item['ground_truth']):
                        num_correct += 1

                n, c = Config.NUM_SAMPLES_PER_TASK, num_correct
                task_pass_k = {k: calculate_pass_at_k(n, c, k) for k in Config.K_VALUES_TO_CALCULATE}

                eval_result = {
                    "task_id": task_id, "reponame": gt_item['reponame'], "condition": cond,
                    "mode": mode, "ground_truth": gt_item['ground_truth'], "predictions": extracted_predictions,
                    "num_correct": c, "num_samples": n, "pass_at_k": task_pass_k
                }
                output_files[cond].write(json.dumps(eval_result, ensure_ascii=False) + '\n')

                summary_group = final_summary[cond]
                summary_group['num_problems'] += 1
                for k, v in task_pass_k.items():
                    summary_group['pass_k'][k] += v

        for f in output_files.values():
            f.close()
            
        for cond in ["original", "rewrite"]:
            summary_data = final_summary[cond]
            num_problems = summary_data['num_problems']
            if num_problems == 0: continue

            row_key = f"{base_model_name}|{cond}"
            found_row = next((row for row in all_results_collector if row.get("row_key") == row_key), None)
            
            if found_row is None:
                found_row = {"Model": base_model_name, "Condition": cond, "row_key": row_key}
                all_results_collector.append(found_row)
            
            final_pass_k = {k: v / num_problems for k, v in summary_data['pass_k'].items()}
            
            for k, val in final_pass_k.items():
                if k in Config.K_VALUES_TO_DISPLAY:
                    column_name = f"{data_size_key}_{mode}_pass@{k}"
                    found_row[column_name] = val

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate all code completion experiments found in a directory.")
    parser.add_argument("--parent_dir", type=str, default=".", help="The parent directory containing all 'experiments_output_*' folders.")
    args = parser.parse_args()

    parent_path = Path(args.parent_dir)
    if not parent_path.is_dir():
        print(f"FATAL: Parent directory not found at '{parent_path}'")
        exit(1)

    experiment_dirs = sorted([d for d in parent_path.glob("experiments_output_*") if d.is_dir()])
    if not experiment_dirs:
        print("No experiment directories with prefix 'experiments_output_' found. Nothing to do.")
        exit(0)
    
    print(f"Found {len(experiment_dirs)} experiment(s) to evaluate:")
    for exp_dir in experiment_dirs: print(f"  - {exp_dir.name}")
    
    all_results = []
    for exp_dir in experiment_dirs:
        evaluate_experiment(exp_dir, all_results)

    print_summary_table(all_results)
    print("\n\n===== Batch Evaluation Script Finished =====")

if __name__ == "__main__":
    main()