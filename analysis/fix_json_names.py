"""
Fix dataset naming inconsistencies in all JSON result files
This ensures consistency when analyze_all_results.py re-runs
"""

import json
from pathlib import Path

def fix_json_file(file_path):
    """Fix dataset naming in a single JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check if metadata exists and has dataset field
        if 'metadata' in data and 'dataset' in data['metadata']:
            old_name = data['metadata']['dataset']
            
            # Standardize smartphone dataset names
            if old_name in ['har_smartphone', 'smartphone', 'smartphone_har']:
                data['metadata']['dataset'] = 'smartphone_har'
                
                # Save the fixed JSON
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✓ Fixed: {file_path.parent.parent.name}/{file_path.parent.name}/{file_path.name}")
                print(f"  Changed '{old_name}' → 'smartphone_har'")
                return True
        
        return False
    
    except Exception as e:
        print(f"✗ Error with {file_path}: {e}")
        return False

def main():
    BASE_PATH = Path(__file__).resolve().parent.parent
    
    print("=" * 80)
    print("FIXING DATASET NAMING IN JSON FILES")
    print("=" * 80)
    print()
    
    # Find all JSON result files
    datasets = ['diabetes', 'heart', 'obesity', 'smartphone', 'sms_spam', 'mamogram', 'sonar', 'thoracic']
    algorithms = ['logistic', 'knn', 'decision_tree', 'svm']
    
    fixed_count = 0
    total_count = 0
    
    for dataset in datasets:
        for algorithm in algorithms:
            algo_folder = BASE_PATH / dataset / algorithm
            if algo_folder.exists():
                for json_file in algo_folder.glob('*sensitivity_results.json'):
                    total_count += 1
                    if fix_json_file(json_file):
                        fixed_count += 1
    
    print()
    print("=" * 80)
    print(f"COMPLETE: Fixed {fixed_count} out of {total_count} JSON files")
    print("=" * 80)
    print()
    print("Next step: Run 'python analyze_all_results.py' to regenerate visualizations")

if __name__ == "__main__":
    main()
