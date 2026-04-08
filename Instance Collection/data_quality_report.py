# -*- coding: utf-8 -*-

import json
import os
import sys
from collections import defaultdict, Counter
from typing import Dict, List, Tuple
from pathlib import Path

class DataQualityReporter:
    """
    Data quality report generator for analyzing the effectiveness of data quality filtering
    """
    
    def __init__(self):
        self.stats = {
            'total_assertions': 0,
            'quality_assertions': 0,
            'filtered_assertions': 0,
            'repos_processed': 0,
            'quality_by_repo': defaultdict(dict),
            'complexity_distribution': Counter(),
            'filter_reasons': Counter()
        }
    
    def analyze_data_file(self, file_path: str) -> Dict[str, any]:
        """
        Analyze quality statistics from a single data file
        """
        if not os.path.exists(file_path):
            return {}
        
        repo_stats = {
            'total_assertions': 0,
            'quality_assertions': 0,
            'filtered_assertions': 0,
            'complexity_scores': [],
            'filter_reasons': Counter()
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        repo_stats['total_assertions'] += 1
                        
                        # If quality analysis info exists, collect statistics
                        if 'quality_analysis' in data:
                            analysis = data['quality_analysis']
                            complexity = analysis.get('complexity_score', 0)
                            repo_stats['complexity_scores'].append(complexity)
                            
                            if analysis.get('is_quality', False):
                                repo_stats['quality_assertions'] += 1
                            else:
                                repo_stats['filtered_assertions'] += 1
                                reason = analysis.get('reason', 'Unknown')
                                repo_stats['filter_reasons'][reason] += 1
                        else:
                            # If no quality analysis, assume it passed the filter
                            repo_stats['quality_assertions'] += 1
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Warning: Failed to analyze file {file_path}: {e}", file=sys.stderr)
            return {}
        
        return repo_stats
    
    def generate_quality_report(self, data_dir: str, output_file: str = None):
        """
        Generate a data quality report
        """
        data_path = Path(data_dir)
        if not data_path.exists() or not data_path.is_dir():
            print(f"Error: Data directory '{data_dir}' does not exist", file=sys.stderr)
            return
        
        # Analyze all data files
        jsonl_files = list(data_path.glob("*.jsonl"))
        if not jsonl_files:
            print(f"Warning: No .jsonl files found in directory '{data_dir}'", file=sys.stderr)
            return
        
        print(f"Analyzing {len(jsonl_files)} data files...")
        
        for file_path in jsonl_files:
            reponame = file_path.stem
            repo_stats = self.analyze_data_file(str(file_path))
            
            if repo_stats:
                self.stats['repos_processed'] += 1
                self.stats['total_assertions'] += repo_stats['total_assertions']
                self.stats['quality_assertions'] += repo_stats['quality_assertions']
                self.stats['filtered_assertions'] += repo_stats['filtered_assertions']
                
                # Record stats per repository
                self.stats['quality_by_repo'][reponame] = repo_stats
                
                # Record complexity distribution
                for score in repo_stats['complexity_scores']:
                    self.stats['complexity_distribution'][score] += 1
                
                # Record filtering reasons
                for reason, count in repo_stats['filter_reasons'].items():
                    self.stats['filter_reasons'][reason] += count
        
        # Generate report
        self._print_report()
        
        # Save detailed report to file
        if output_file:
            self._save_detailed_report(output_file)
    
    def _print_report(self):
        """
        Print quality report to console
        """
        print("\n" + "=" * 80)
        print("Data Quality Analysis Report")
        print("=" * 80)
        
        total = self.stats['total_assertions']
        quality = self.stats['quality_assertions']
        filtered = self.stats['filtered_assertions']
        
        print(f"\nOverall Statistics:")
        print(f"  - Repositories processed: {self.stats['repos_processed']}")
        print(f"  - Total assertions: {total}")
        print(f"  - High-quality assertions: {quality} ({quality/total*100:.1f}%)")
        print(f"  - Filtered-out assertions: {filtered} ({filtered/total*100:.1f}%)")
        
        if self.stats['complexity_distribution']:
            print(f"\nComplexity Distribution:")
            for score in sorted(self.stats['complexity_distribution'].keys()):
                count = self.stats['complexity_distribution'][score]
                print(f"  - Complexity {score}: {count} assertions")
        
        if self.stats['filter_reasons']:
            print(f"\nFiltering Reasons:")
            for reason, count in self.stats['filter_reasons'].most_common():
                print(f"  - {reason}: {count} assertions")
        
        print(f"\nPer-repository Quality Stats:")
        for reponame, repo_stats in sorted(self.stats['quality_by_repo'].items()):
            repo_total = repo_stats['total_assertions']
            repo_quality = repo_stats['quality_assertions']
            quality_rate = repo_quality / repo_total * 100 if repo_total > 0 else 0
            print(f"  - {reponame}: {repo_quality}/{repo_total} ({quality_rate:.1f}%)")
        
        print("=" * 80)
    
    def _save_detailed_report(self, output_file: str):
        """
        Save detailed report to a file
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            print(f"\nDetailed report saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Could not save detailed report to {output_file}: {e}", file=sys.stderr)

def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Data quality analysis report generator")
    parser.add_argument('data_dir', help="Directory containing .jsonl data files")
    parser.add_argument('--output', '-o', help="Output path for detailed JSON report")
    
    args = parser.parse_args()
    
    reporter = DataQualityReporter()
    reporter.generate_quality_report(args.data_dir, args.output)

if __name__ == '__main__':
    main()