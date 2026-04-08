import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Set

def process_runtime_value(runtime_info: Dict[str, Any], original_gt: str) -> List[str]:
    """
    Extract new ground truth strings based on the type and value of runtime information, along with defined rules.
    """
    if not runtime_info or runtime_info.get("status") != "success":
        return []

    value = runtime_info.get("value")
    type_str = runtime_info.get("type", "")
    new_truths: Set[str] = set()

    if type_str == "<class 'NoneType'>":
        return []
    
    # Enhanced handling for ApproxScalar
    elif type_str == "<class '_pytest.python_api.ApproxScalar'>":
        # Strategy 1: Try to extract numeric literals from the original GT string first
        match = re.search(r"pytest\.approx\(([^,)]+)", original_gt)
        if match:
            num_str = match.group(1).strip()
            try:
                # If successful, it means the argument is a literal like "1", "0.5"
                num = float(num_str)
                if num.is_integer():
                    new_truths.add(str(int(num)))
                    new_truths.add(str(float(num)))
                else:
                    new_truths.add(str(num))
                # After successfully parsing the literal, return immediately
                return list(new_truths)
            except ValueError:
                # If parsing fails (e.g., the argument is 'test[1]'), ignore and fall through
                # to the fallback strategy below.
                pass

        # Fallback Strategy 2: Parse the runtime value string (e.g., "-45.6 \u00b1 4.6e-05")
        if isinstance(value, str) and '\u00b1' in value:
            # Extract the part before the '±' symbol
            num_part_str = value.split('\u00b1')[0].strip()
            try:
                # Convert the extracted part to a float and add it
                float_val = float(num_part_str)
                new_truths.add(str(float_val))
            except ValueError:
                # If parsing fails, ignore
                pass
        
        return list(new_truths)

    elif type_str in ["<class 'numpy.int32'>", "<class 'numpy.int64'>", "<class 'numpy.uint8'>"]:
        new_truths.add(str(value))
        return list(new_truths)
    elif type_str == "<class 'tuple'>":
        if isinstance(value, list):
            new_truths.add(str(tuple(value)))
        return list(new_truths)
    elif type_str == "<class 'str'>":
        new_truths.add(repr(value))
        return list(new_truths)
    else:
        new_truths.add(str(value))
        return list(new_truths)

def main():
    parser = argparse.ArgumentParser(description="Enhance Ground Truth dataset.")
    parser.add_argument("--data_dir", type=str, default="../data_collection_align", help="Path to the directory of the original dataset.")
    parser.add_argument("--runtime_dir", type=str, default="../groundtruth_collection", help="Path to the directory containing runtime Ground Truth data.")
    parser.add_argument("--output_dir", type=str, default="../Data_RepoReasoner", help="Path to the directory for storing the enhanced dataset.")
    args = parser.parse_args()

    DATA_DIR = Path(args.data_dir)
    RUNTIME_DIR = Path(args.runtime_dir)
    OUTPUT_DIR = Path(args.output_dir)

    if not DATA_DIR.is_dir() or not RUNTIME_DIR.is_dir():
        print(f"Error: Please ensure input directories '{DATA_DIR}' and '{RUNTIME_DIR}' exist.")
        return

    conditions = ["original", "rewrite"]
    for condition in conditions:
        print("\n" + "#"*60)
        print(f"# Processing '{condition.upper()}' data")
        print("#"*60)

        runtime_file = RUNTIME_DIR / f"{condition}.jsonl"
        source_data_dir = DATA_DIR / condition
        output_data_dir = OUTPUT_DIR / condition

        if not runtime_file.exists():
            print(f"Warning: Runtime file '{runtime_file}' not found, skipping '{condition}'.")
            continue
        if not source_data_dir.is_dir():
            print(f"Warning: Source data directory '{source_data_dir}' not found, skipping '{condition}'.")
            continue

        print(f"Loading runtime data from '{runtime_file}'...")
        runtime_map: Dict[str, Dict] = {}
        with open(runtime_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                runtime_map[data['task_id']] = data
        print(f"Load complete. Found {len(runtime_map)} runtime records for '{condition}'.")

        source_files = list(source_data_dir.glob("**/*.jsonl"))
        if not source_files:
            print(f"Warning: No .jsonl files found in '{source_data_dir}'.")
            continue

        output_data_dir.mkdir(parents=True, exist_ok=True)
        print(f"Processing files from '{source_data_dir}' and writing to '{output_data_dir}'...")

        for source_file_path in source_files:
            relative_path = source_file_path.relative_to(source_data_dir)
            output_file_path = output_data_dir / relative_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"  -> Processing {source_file_path} -> {output_file_path}")

            with open(source_file_path, 'r', encoding='utf-8') as f_in, \
                 open(output_file_path, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    task_data = json.loads(line)
                    task_id = task_data['task_id']
                    original_gt = task_data['ground_truth']
                    
                    all_gts: Set[str] = {original_gt}
                    
                    if task_id in runtime_map:
                        runtime_info = runtime_map[task_id].get("runtime_ground_truth", {})
                        new_truths = process_runtime_value(runtime_info, original_gt)
                        
                        for truth in new_truths:
                            all_gts.add(truth)
                    
                    task_data['ground_truth'] = sorted(list(all_gts))
                    f_out.write(json.dumps(task_data) + '\n')

    print("\nAll processing completed!")
    print(f"Enhanced dataset has been saved to: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()