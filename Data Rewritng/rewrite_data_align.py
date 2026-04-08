import os
import json
from collections import defaultdict
import sys

def group_data_by_function(file_path: str) -> dict:
    """
    Read a .jsonl file and group its contents by (testname, classname, funcname).
    """
    data_map = defaultdict(list)
    if not os.path.exists(file_path):
        return data_map
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                key = (
                    data.get('testname'),
                    data.get('classname'), 
                    data.get('funcname')
                )
                if key[0] and key[2]:
                    data_map[key].append(data)
            except (json.JSONDecodeError, KeyError):
                continue
    return data_map

def align_datasets_final():
    """
    Align datasets based on the minimum number of assertions in each test function before and after rewrite,
    and enforce consistent task_id and testpath between original and rewrite data.
    """
    current_dir = os.getcwd()
    upper_dir = os.path.dirname(current_dir)

    # --- Define directory paths ---
    source_dir_original = os.path.join(upper_dir, 'data_collection')
    source_dir_rewrite = os.path.join(upper_dir, 'data_collection_from_rewrites_filtered')
    
    align_base_dir = os.path.join(upper_dir, 'data_collection_align')
    output_dir_original = os.path.join(align_base_dir, 'original')
    output_dir_rewrite = os.path.join(align_base_dir, 'rewrite')

    # --- Setup ---
    print("--- Starting final data alignment (unifying task_id and testpath) ---")
    if not os.path.isdir(source_dir_original) or not os.path.isdir(source_dir_rewrite):
        print(f"Error: Please ensure source directories 'data_collection' and 'data_collection_from_rewrites_filtered' both exist.", file=sys.stderr)
        return

    os.makedirs(output_dir_original, exist_ok=True)
    os.makedirs(output_dir_rewrite, exist_ok=True)
    
    print(f"Original data source: {source_dir_original}")
    print(f"Rewritten data source: {source_dir_rewrite}")
    print(f"Aligned output to: {output_dir_original} and {output_dir_rewrite}")
    print("-" * 70)

    total_original_lines_before, total_rewrite_lines_before, total_aligned_lines = 0, 0, 0
    original_files = [f for f in os.listdir(source_dir_original) if f.endswith('.jsonl')]

    for filename in original_files:
        reponame = filename.replace('.jsonl', '')
        print(f"\n--- Processing repository: {reponame} ---")

        original_file_path = os.path.join(source_dir_original, filename)
        rewrite_file_path = os.path.join(source_dir_rewrite, filename)

        if not os.path.exists(rewrite_file_path):
            print(f"  Warning: Corresponding file '{filename}' not found in rewritten data directory. Skipped."); continue

        original_data_map = group_data_by_function(original_file_path)
        rewrite_data_map = group_data_by_function(rewrite_file_path)

        original_line_count = sum(len(v) for v in original_data_map.values())
        rewrite_line_count = sum(len(v) for v in rewrite_data_map.values())
        total_original_lines_before += original_line_count
        total_rewrite_lines_before += rewrite_line_count

        print(f"  Source file line count: Original={original_line_count}, Rewritten={rewrite_line_count}")

        aligned_count_for_repo = 0
        output_original_file, output_rewrite_file = os.path.join(output_dir_original, filename), os.path.join(output_dir_rewrite, filename)

        with open(output_original_file, 'w', encoding='utf-8') as f_out_orig, \
             open(output_rewrite_file, 'w', encoding='utf-8') as f_out_rewrite:

            common_keys_set = original_data_map.keys() & rewrite_data_map.keys()
            
            common_keys = sorted(
                list(common_keys_set), 
                key=lambda k: (str(k[0]), str(k[1]), str(k[2]))
            )

            print(f"  Found {len(common_keys)} common test functions/methods for alignment.")

            for key in common_keys:
                original_assertions, rewrite_assertions = original_data_map[key], rewrite_data_map[key]
                num_to_keep = min(len(original_assertions), len(rewrite_assertions))
                
                for i in range(num_to_keep):
                    original_item, rewrite_item = original_assertions[i], rewrite_assertions[i]
                    aligned_rewrite_item = rewrite_item.copy()
                    aligned_rewrite_item['task_id'] = original_item['task_id']
                    aligned_rewrite_item['testpath'] = original_item['testpath']
                    f_out_orig.write(json.dumps(original_item, ensure_ascii=False) + '\n')
                    f_out_rewrite.write(json.dumps(aligned_rewrite_item, ensure_ascii=False) + '\n')
                aligned_count_for_repo += num_to_keep
        
        total_aligned_lines += aligned_count_for_repo
        print(f"  ✅ Alignment completed: Both output files have written {aligned_count_for_repo} lines of one-to-one matched data.")

    print("\n" + "=" * 70 + "\n--- Overall Alignment Report ---")
    print(f"\nTotal lines before alignment:\n  - Original data: {total_original_lines_before}\n  - Rewritten data: {total_rewrite_lines_before}")
    print(f"\nTotal lines after alignment:\n  - Total lines in both 'original' and 'rewrite' directories: {total_aligned_lines}")
    print("=" * 70 + "\n\n--- All tasks completed ---")

if __name__ == '__main__':
    align_datasets_final()