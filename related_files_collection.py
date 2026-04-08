import json
import os
from pathlib import Path
import sys

# You need to install this library first: pip install rank-bm25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("Error: Dependency 'rank-bm25' is not installed. Please run 'pip install rank-bm25' to install it.", file=sys.stderr)
    sys.exit(1)

# Cache corpus for each repository to avoid rebuilding
CORPUS_CACHE = {}

# Define a set of directory names to exclude globally, for easier management
EXCLUDED_DIR_NAMES = {
    # User-specified
    'venv', 'node_modules', 'build', 'dist', 'site-packages', 'tests', 'test', 'testing',
    # Common irrelevant directories
    '.venv', '.git', '.hg', 'docs', 'examples', 'samples', 'scripts'
}

def create_corpus_for_repo(repo_path: Path):
    """
    Build a corpus for a single repository, excluding test and irrelevant files/directories.
    Returns a list of file paths and a list of tokenized content.
    """
    repo_key = str(repo_path)
    if repo_key in CORPUS_CACHE:
        return CORPUS_CACHE[repo_key]

    print(f"  - Building corpus for repository '{repo_path.name}'...")
    doc_paths = []
    tokenized_corpus = []

    all_py_files = list(repo_path.rglob("*.py"))

    for file_path in all_py_files:
        path_parts = {p.lower() for p in file_path.parts}

        if not EXCLUDED_DIR_NAMES.isdisjoint(path_parts):
            continue
        
        if file_path.name in ("__init__.py", "setup.py", "conftest.py"):
            continue
        
        try:
            content = file_path.read_text(encoding="utf-8")
            tokenized_content = content.split()
            
            if tokenized_content:
                doc_paths.append(file_path.relative_to(repo_path).as_posix())
                tokenized_corpus.append(tokenized_content)
        except Exception:
            continue
    
    print(f"  - Corpus built successfully, containing {len(doc_paths)} source files.")
    result = (doc_paths, tokenized_corpus)
    CORPUS_CACHE[repo_key] = result
    return result

def analyze_with_bm25(input_jsonl, output_jsonl, repos_base_dir):
    """
    Process a single jsonl file using BM25, with caching for repeated queries.
    """
    reponame = Path(input_jsonl).stem
    repo_root = Path(repos_base_dir) / "python_repos" / reponame

    if not repo_root.is_dir():
        print(f"Error: Repository directory '{repo_root}' for '{reponame}' not found.", file=sys.stderr)
        return

    corpus_paths, tokenized_corpus = create_corpus_for_repo(repo_root)
    
    if not corpus_paths:
        print(f"Warning: Corpus for repository '{reponame}' is empty, skipping processing.", file=sys.stderr)
        return
        
    bm25 = BM25Okapi(tokenized_corpus)

    # ✨ Cache BM25 retrieval results for each testpath
    # Key is testpath, value is the list of retrieved relevant files
    query_results_cache = {}

    with open(input_jsonl, 'r', encoding='utf-8') as infile, \
         open(output_jsonl, 'w', encoding='utf-8') as outfile:
        
        print(f"  - Starting to process entries in {reponame}...")
        for line in infile:
            try:
                data = json.loads(line)
                testpath = data.get("testpath")

                if not testpath:
                    continue

                # ✨ Core logic: Check cache
                if testpath in query_results_cache:
                    # If result for this test file is already cached, use it directly
                    ranked_files = query_results_cache[testpath]
                else:
                    # If encountering this test file for the first time, perform retrieval
                    # print(f"    - Performing BM25 retrieval for new file '{testpath}'...") # (Optional) Debug print
                    query_test_path = repo_root / testpath

                    if not query_test_path.is_file():
                        ranked_files = []
                    else:
                        query_content = query_test_path.read_text(encoding="utf-8")
                        tokenized_query = query_content.split()
                        ranked_files = bm25.get_top_n(tokenized_query, corpus_paths, n=len(corpus_paths))

                    # ✨ Store the new result in cache for future use
                    query_results_cache[testpath] = ranked_files

                # Construct and write output data
                output_data = {
                    "task_id": data.get("task_id"),
                    "reponame": data.get("reponame"),
                    "testpath": data.get("testpath"),
                    "testname": data.get("testname"),
                    "funcname": data.get("funcname"),
                    "related_files_rank": ranked_files
                }
                
                outfile.write(json.dumps(output_data) + '\n')

            except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
                # Print error message for debugging, but continue processing
                print(f"    - Warning: Error processing line (error: {e}), skipped. Line content: {line.strip()}", file=sys.stderr)
                continue

def main():
    """
    Main logic for batch processing.
    """
    # ✨ Suggestion: Use absolute paths or paths relative to the script for robustness
    script_dir = Path(__file__).parent
    REPOS_BASE_DIR = script_dir
    SOURCE_DATA_DIR = script_dir / 'Data_RepoReasoner' / 'original'
    OUTPUT_DIR = script_dir / 'output_with_bm25_rank'
    
    if not SOURCE_DATA_DIR.is_dir():
        print(f"Error: Input directory '{SOURCE_DATA_DIR}' does not exist.", file=sys.stderr)
        return
        
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    files_to_process = [f for f in os.listdir(SOURCE_DATA_DIR) if f.endswith('.jsonl')]
    
    print(f"Found {len(files_to_process)} files to process, output will be saved to '{OUTPUT_DIR}'...")

    for filename in files_to_process:
        input_path = SOURCE_DATA_DIR / filename
        output_path = OUTPUT_DIR / filename
        
        print(f"\n[+] Processing: {filename}")
        analyze_with_bm25(str(input_path), str(output_path), str(REPOS_BASE_DIR))
        print(f"[✓] Completed: Results saved to {output_path}")
        
    print("\nAll files processed!")

if __name__ == '__main__':
    main()