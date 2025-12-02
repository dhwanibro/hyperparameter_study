"""
Hyperparameter Sensitivity Analysis
Analyzes how different datasets and algorithms react to hyperparameter tuning
"""

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: FIND ALL RESULT FILES
# ============================================================================

def find_result_files(base_path):
    """
    Find all sensitivity result txt files in the project
    Returns: list of file paths
    """
    result_files = []
    base_path = Path(base_path)
    
    # Look in each dataset folder
    datasets = ['diabetes', 'heart', 'obesity', 'smartphone', 'sms_spam']
    algorithms = ['logistic', 'knn', 'decision_tree', 'svm']
    
    for dataset in datasets:
        for algorithm in algorithms:
            # Construct possible paths
            algo_folder = base_path / dataset / algorithm
            if algo_folder.exists():
                # Look for any txt file with 'sensitivity' or 'results' in name
                for txt_file in algo_folder.glob('*sensitivity*.txt'):
                    result_files.append(txt_file)
                for txt_file in algo_folder.glob('*results*.txt'):
                    if 'sensitivity' in txt_file.name:
                        result_files.append(txt_file)
    
    return result_files

# ============================================================================
# STEP 2: PARSE TEXT FILES TO EXTRACT DATA
# ============================================================================

def parse_logistic_file(file_path):
    """Parse logistic regression results"""
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract C values and their metrics
    c_pattern = r'C = ([\d.]+):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(c_pattern, content):
        data.append({
            'param_name': 'C',
            'param_value': float(match.group(1)),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    # Extract penalty types (L1, L2)
    penalty_pattern = r'(L1|L2):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(penalty_pattern, content):
        data.append({
            'param_name': 'penalty',
            'param_value': match.group(1),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    return data

def parse_knn_file(file_path):
    """Parse KNN results"""
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract k values
    k_pattern = r'k = (\d+): Train Acc = ([\d.]+), Test Acc = ([\d.]+)'
    for match in re.finditer(k_pattern, content):
        data.append({
            'param_name': 'k',
            'param_value': int(match.group(1)),
            'accuracy': float(match.group(3)),
            'precision': None,
            'recall': None,
            'f1': None,
            'auc': None
        })
    
    # Extract weights
    weights_pattern = r'(uniform|distance): Train Acc = [\d.]+, Test Acc = ([\d.]+)'
    for match in re.finditer(weights_pattern, content):
        data.append({
            'param_name': 'weights',
            'param_value': match.group(1),
            'accuracy': float(match.group(2)),
            'precision': None,
            'recall': None,
            'f1': None,
            'auc': None
        })
    
    # Extract metrics
    metric_pattern = r'(euclidean|manhattan): Train Acc = [\d.]+, Test Acc = ([\d.]+)'
    for match in re.finditer(metric_pattern, content):
        data.append({
            'param_name': 'metric',
            'param_value': match.group(1),
            'accuracy': float(match.group(2)),
            'precision': None,
            'recall': None,
            'f1': None,
            'auc': None
        })
    
    return data

def parse_decision_tree_file(file_path):
    """Parse decision tree results"""
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract max_depth
    depth_pattern = r'max_depth = (\w+):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(depth_pattern, content):
        depth_val = None if match.group(1) == 'None' else int(match.group(1))
        data.append({
            'param_name': 'max_depth',
            'param_value': depth_val,
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    # Extract min_samples_split
    split_pattern = r'min_samples_split = (\d+):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(split_pattern, content):
        data.append({
            'param_name': 'min_samples_split',
            'param_value': int(match.group(1)),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    # Extract min_samples_leaf
    leaf_pattern = r'min_samples_leaf = (\d+):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(leaf_pattern, content):
        data.append({
            'param_name': 'min_samples_leaf',
            'param_value': int(match.group(1)),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    # Extract criterion
    criterion_pattern = r'(gini|entropy|log_loss):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(criterion_pattern, content):
        data.append({
            'param_name': 'criterion',
            'param_value': match.group(1),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    return data

def parse_svm_file(file_path):
    """Parse SVM results"""
    data = []
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract C values
    c_pattern = r'C = ([\d.]+):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(c_pattern, content):
        data.append({
            'param_name': 'C',
            'param_value': float(match.group(1)),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    # Extract kernel types
    kernel_pattern = r'(linear|rbf|poly|sigmoid):\s+Test Metrics: Acc=([\d.]+), Prec=([\d.]+), Rec=([\d.]+), F1=([\d.]+), AUC=([\d.]+)'
    for match in re.finditer(kernel_pattern, content):
        data.append({
            'param_name': 'kernel',
            'param_value': match.group(1),
            'accuracy': float(match.group(2)),
            'precision': float(match.group(3)),
            'recall': float(match.group(4)),
            'f1': float(match.group(5)),
            'auc': float(match.group(6))
        })
    
    return data

def parse_file(file_path):
    """
    Main parser - determines algorithm type and calls appropriate parser
    """
    file_path = Path(file_path)
    
    # Determine algorithm from path
    if 'logistic' in str(file_path):
        algorithm = 'logistic_regression'
        data = parse_logistic_file(file_path)
    elif 'knn' in str(file_path):
        algorithm = 'knn'
        data = parse_knn_file(file_path)
    elif 'decision_tree' in str(file_path):
        algorithm = 'decision_tree'
        data = parse_decision_tree_file(file_path)
    elif 'svm' in str(file_path):
        algorithm = 'svm'
        data = parse_svm_file(file_path)
    else:
        return None
    
    # Determine dataset from path
    for dataset in ['diabetes', 'heart', 'obesity', 'smartphone', 'sms_spam']:
        if dataset in str(file_path):
            dataset_name = dataset
            break
    else:
        dataset_name = 'unknown'
    
    # Add metadata to each record
    for record in data:
        record['dataset'] = dataset_name
        record['algorithm'] = algorithm
    
    return data

# ============================================================================
# STEP 3: CALCULATE SENSITIVITY METRICS
# ============================================================================

def calculate_sensitivity(df, metric='accuracy'):
    """
    Calculate sensitivity for each hyperparameter
    Sensitivity = variance of metric values when parameter changes
    """
    sensitivity_results = []
    
    # Group by dataset, algorithm, and parameter
    for (dataset, algorithm, param), group in df.groupby(['dataset', 'algorithm', 'param_name']):
        # Get metric values
        values = group[metric].dropna()
        
        if len(values) > 1:  # Need at least 2 values to calculate variance
            sensitivity = {
                'dataset': dataset,
                'algorithm': algorithm,
                'parameter': param,
                'variance': values.var(),
                'std_dev': values.std(),
                'range': values.max() - values.min(),
                'mean': values.mean(),
                'min': values.min(),
                'max': values.max(),
                'cv': values.std() / values.mean() if values.mean() != 0 else 0  # coefficient of variation
            }
            sensitivity_results.append(sensitivity)
    
    return pd.DataFrame(sensitivity_results)

# ============================================================================
# STEP 4: CREATE VISUALIZATIONS
# ============================================================================

def plot_dataset_sensitivity(df, output_dir):
    """
    Plot: Which dataset is most sensitive overall?
    """
    # Calculate average variance per dataset
    dataset_sens = df.groupby('dataset')['variance'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    dataset_sens.plot(kind='bar', color='steelblue')
    plt.title('Dataset Sensitivity to Hyperparameter Changes', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Average Variance', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: dataset_sensitivity.png")

def plot_algorithm_sensitivity(df, output_dir):
    """
    Plot: Which algorithm is most sensitive overall?
    """
    # Calculate average variance per algorithm
    algo_sens = df.groupby('algorithm')['variance'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    algo_sens.plot(kind='bar', color='coral')
    plt.title('Algorithm Sensitivity to Hyperparameter Changes', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Average Variance', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'algorithm_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: algorithm_sensitivity.png")

def plot_parameter_sensitivity(df, output_dir):
    """
    Plot: Which hyperparameters are most important per algorithm?
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    algorithms = ['logistic_regression', 'knn', 'decision_tree', 'svm']
    
    for idx, algorithm in enumerate(algorithms):
        ax = axes[idx // 2, idx % 2]
        
        # Get data for this algorithm
        algo_data = df[df['algorithm'] == algorithm]
        
        if len(algo_data) > 0:
            # Average variance per parameter
            param_sens = algo_data.groupby('parameter')['variance'].mean().sort_values(ascending=False)
            
            param_sens.plot(kind='barh', ax=ax, color='mediumseagreen')
            ax.set_title(f'{algorithm.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Average Variance', fontsize=10)
            ax.set_ylabel('Parameter', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
    
    plt.suptitle('Hyperparameter Sensitivity by Algorithm', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: parameter_sensitivity.png")

def plot_heatmap(df, output_dir):
    """
    Plot: Heatmap of Dataset × Algorithm sensitivity
    """
    # Create pivot table: dataset vs algorithm
    pivot = df.pivot_table(values='variance', index='dataset', columns='algorithm', aggfunc='mean')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlOrRd', cbar_kws={'label': 'Average Variance'})
    plt.title('Sensitivity Heatmap: Dataset × Algorithm', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / 'sensitivity_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: sensitivity_heatmap.png")

def plot_comparison_across_datasets(df, output_dir):
    """
    Plot: How does the same parameter behave across datasets?
    """
    # Focus on common parameters
    common_params = df.groupby('parameter').size().sort_values(ascending=False).head(5).index
    
    fig, axes = plt.subplots(len(common_params), 1, figsize=(12, 3*len(common_params)))
    
    if len(common_params) == 1:
        axes = [axes]
    
    for idx, param in enumerate(common_params):
        ax = axes[idx]
        
        # Get data for this parameter
        param_data = df[df['parameter'] == param]
        
        # Plot variance by dataset
        pivot = param_data.pivot_table(values='variance', index='dataset', columns='algorithm', aggfunc='mean')
        pivot.plot(kind='bar', ax=ax)
        
        ax.set_title(f'Parameter: {param}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Dataset', fontsize=10)
        ax.set_ylabel('Variance', fontsize=10)
        ax.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Parameter Behavior Across Datasets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: parameter_comparison.png")

# ============================================================================
# STEP 5: GENERATE SUMMARY REPORT
# ============================================================================

def generate_summary_report(df, output_dir):
    """
    Generate text summary of findings
    """
    report = []
    report.append("=" * 80)
    report.append("HYPERPARAMETER SENSITIVITY ANALYSIS - SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # 1. Most/Least sensitive datasets
    dataset_sens = df.groupby('dataset')['variance'].mean().sort_values(ascending=False)
    report.append("1. DATASET SENSITIVITY RANKING")
    report.append("-" * 80)
    for i, (dataset, var) in enumerate(dataset_sens.items(), 1):
        report.append(f"   {i}. {dataset:20s} - Average Variance: {var:.6f}")
    report.append(f"\n   Most Sensitive:  {dataset_sens.index[0]}")
    report.append(f"   Least Sensitive: {dataset_sens.index[-1]}")
    report.append("")
    
    # 2. Most/Least sensitive algorithms
    algo_sens = df.groupby('algorithm')['variance'].mean().sort_values(ascending=False)
    report.append("2. ALGORITHM SENSITIVITY RANKING")
    report.append("-" * 80)
    for i, (algo, var) in enumerate(algo_sens.items(), 1):
        report.append(f"   {i}. {algo:25s} - Average Variance: {var:.6f}")
    report.append(f"\n   Most Sensitive:  {algo_sens.index[0]}")
    report.append(f"   Least Sensitive: {algo_sens.index[-1]}")
    report.append("")
    
    # 3. Most important parameters per algorithm
    report.append("3. MOST IMPORTANT HYPERPARAMETERS PER ALGORITHM")
    report.append("-" * 80)
    for algorithm in df['algorithm'].unique():
        algo_data = df[df['algorithm'] == algorithm]
        param_sens = algo_data.groupby('parameter')['variance'].mean().sort_values(ascending=False)
        
        report.append(f"\n   {algorithm.upper().replace('_', ' ')}:")
        for i, (param, var) in enumerate(param_sens.items(), 1):
            report.append(f"      {i}. {param:20s} - Variance: {var:.6f}")
    report.append("")
    
    # 4. Dataset-Algorithm combinations
    report.append("4. MOST SENSITIVE DATASET-ALGORITHM COMBINATIONS")
    report.append("-" * 80)
    combo_sens = df.groupby(['dataset', 'algorithm'])['variance'].mean().sort_values(ascending=False).head(10)
    for i, ((dataset, algo), var) in enumerate(combo_sens.items(), 1):
        report.append(f"   {i}. {dataset:15s} + {algo:25s} = {var:.6f}")
    report.append("")
    
    # 5. Key insights
    report.append("5. KEY INSIGHTS")
    report.append("-" * 80)
    
    # Which parameters are universally important?
    universal_params = df.groupby('parameter')['variance'].mean().sort_values(ascending=False).head(3)
    report.append("   Universally Important Parameters:")
    for param, var in universal_params.items():
        report.append(f"      - {param}: avg variance = {var:.6f}")
    
    # Which are dataset-specific?
    report.append("\n   Dataset-Specific Behavior:")
    for dataset in df['dataset'].unique():
        dataset_data = df[df['dataset'] == dataset]
        top_param = dataset_data.groupby('parameter')['variance'].mean().idxmax()
        top_var = dataset_data.groupby('parameter')['variance'].mean().max()
        report.append(f"      - {dataset}: most sensitive to '{top_param}' (var={top_var:.6f})")
    
    report.append("")
    report.append("=" * 80)
    
    # Write to file
    with open(output_dir / 'summary_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: summary_report.txt")
    
    # Also print to console
    print("\n" + '\n'.join(report))

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    BASE_PATH = Path('/Users/dhwanibalchandani/Developer/ug3/ml_project')
    OUTPUT_DIR = BASE_PATH / 'analysis'
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    print()
    
    # Step 1: Find all result files
    print("Step 1: Finding result files...")
    result_files = find_result_files(BASE_PATH)
    print(f"Found {len(result_files)} result files")
    print()
    
    # Step 2: Parse all files
    print("Step 2: Parsing files...")
    all_data = []
    for file_path in result_files:
        print(f"  - {file_path.parent.parent.name}/{file_path.parent.name}/{file_path.name}")
        parsed_data = parse_file(file_path)
        if parsed_data:
            all_data.extend(parsed_data)
    
    # Create DataFrame
    df_raw = pd.DataFrame(all_data)
    print(f"\nParsed {len(df_raw)} data points")
    print()
    
    # Save raw data
    df_raw.to_csv(OUTPUT_DIR / 'raw_data.csv', index=False)
    print(f"✓ Saved: raw_data.csv")
    print()
    
    # Step 3: Calculate sensitivity metrics
    print("Step 3: Calculating sensitivity metrics...")
    df_sensitivity = calculate_sensitivity(df_raw, metric='accuracy')
    
    # Save sensitivity data
    df_sensitivity.to_csv(OUTPUT_DIR / 'sensitivity_metrics.csv', index=False)
    print(f"✓ Saved: sensitivity_metrics.csv")
    print()
    
    # Step 4: Create visualizations
    print("Step 4: Creating visualizations...")
    plot_dataset_sensitivity(df_sensitivity, OUTPUT_DIR)
    plot_algorithm_sensitivity(df_sensitivity, OUTPUT_DIR)
    plot_parameter_sensitivity(df_sensitivity, OUTPUT_DIR)
    plot_heatmap(df_sensitivity, OUTPUT_DIR)
    plot_comparison_across_datasets(df_sensitivity, OUTPUT_DIR)
    print()
    
    # Step 5: Generate summary report
    print("Step 5: Generating summary report...")
    generate_summary_report(df_sensitivity, OUTPUT_DIR)
    print()
    
    print("=" * 80)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    main()
