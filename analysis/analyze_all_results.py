"""
Comprehensive Hyperparameter Sensitivity Analysis
Analyzes standardized JSON results from all datasets and algorithms
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style for better looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# STEP 1: FIND AND LOAD ALL JSON RESULT FILES
# ============================================================================

def find_json_result_files(base_path):
    """
    Find all *_sensitivity_results.json files in the project
    Returns: list of file paths
    """
    result_files = []
    base_path = Path(base_path)
    
    # Look in each dataset folder
    datasets = ['diabetes', 'heart', 'obesity', 'smartphone', 'sms_spam', 'mamogram', 'sonar', 'thoracic']
    algorithms = ['logistic', 'knn', 'decision_tree', 'svm']
    
    for dataset in datasets:
        for algorithm in algorithms:
            algo_folder = base_path / dataset / algorithm
            if algo_folder.exists():
                # Look for JSON files with 'sensitivity_results' in name
                for json_file in algo_folder.glob('*sensitivity_results.json'):
                    result_files.append(json_file)
    
    return result_files

def load_json_results(file_path):
    """
    Load and parse a single JSON result file
    Returns: list of experiment dictionaries with metadata
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract metadata
        metadata = data.get('metadata', {})
        dataset = metadata.get('dataset', 'unknown')
        algorithm = metadata.get('algorithm', 'unknown')
        
        # Extract experiments and add metadata to each
        experiments = []
        for exp in data.get('experiments', []):
            exp_data = {
                'dataset': dataset,
                'algorithm': algorithm,
                'param_name': exp.get('param_name'),
                'param_value': exp.get('param_value'),
                **exp.get('metrics', {}),
                'additional_info': exp.get('additional_info', {})
            }
            experiments.append(exp_data)
        
        return experiments
    
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []

# ============================================================================
# STEP 2: CALCULATE SENSITIVITY METRICS
# ============================================================================

def calculate_sensitivity_metrics(df):
    """
    Calculate various sensitivity metrics for each hyperparameter
    Metrics include variance, range, coefficient of variation, etc.
    """
    sensitivity_results = []
    
    # Group by dataset, algorithm, and parameter
    for (dataset, algorithm, param), group in df.groupby(['dataset', 'algorithm', 'param_name']):
        # For each metric (accuracy, f1, etc.)
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            if metric in group.columns:
                values = group[metric].dropna()
                
                if len(values) > 1:  # Need at least 2 values
                    sensitivity = {
                        'dataset': dataset,
                        'algorithm': algorithm,
                        'parameter': param,
                        'metric': metric,
                        'variance': float(values.var()),
                        'std_dev': float(values.std()),
                        'range': float(values.max() - values.min()),
                        'mean': float(values.mean()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'cv': float(values.std() / values.mean()) if values.mean() != 0 else 0,  # coefficient of variation
                        'n_values': len(values)
                    }
                    sensitivity_results.append(sensitivity)
    
    return pd.DataFrame(sensitivity_results)

# ============================================================================
# STEP 3: CREATE VISUALIZATIONS
# ============================================================================

def plot_dataset_sensitivity(df, output_dir, metric='accuracy'):
    """
    Plot: Which dataset is most sensitive overall?
    """
    # Filter by metric
    metric_df = df[df['metric'] == metric]
    
    # Calculate average variance per dataset
    dataset_sens = metric_df.groupby('dataset')['variance'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(dataset_sens)))
    dataset_sens.plot(kind='bar', color=colors)
    plt.title(f'Dataset Sensitivity to Hyperparameter Changes ({metric.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Average Variance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'dataset_sensitivity_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: dataset_sensitivity_{metric}.png")

def plot_algorithm_sensitivity(df, output_dir, metric='accuracy'):
    """
    Plot: Which algorithm is most sensitive overall?
    """
    # Filter by metric
    metric_df = df[df['metric'] == metric]
    
    # Calculate average variance per algorithm
    algo_sens = metric_df.groupby('algorithm')['variance'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    algo_sens.plot(kind='bar', color=colors[:len(algo_sens)])
    plt.title(f'Algorithm Sensitivity to Hyperparameter Changes ({metric.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Average Variance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'algorithm_sensitivity_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: algorithm_sensitivity_{metric}.png")

def plot_parameter_sensitivity_by_algorithm(df, output_dir, metric='accuracy'):
    """
    Plot: Which hyperparameters are most important per algorithm?
    """
    # Filter by metric
    metric_df = df[df['metric'] == metric]
    
    algorithms = metric_df['algorithm'].unique()
    n_algos = len(algorithms)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, algorithm in enumerate(algorithms[:4]):  # Max 4 algorithms
        ax = axes[idx]
        
        # Get data for this algorithm
        algo_data = metric_df[metric_df['algorithm'] == algorithm]
        
        if len(algo_data) > 0:
            # Average variance per parameter
            param_sens = algo_data.groupby('parameter')['variance'].mean().sort_values(ascending=False)
            
            param_sens.plot(kind='barh', ax=ax, color='mediumseagreen')
            ax.set_title(f'{algorithm.replace("_", " ").title()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Average Variance', fontsize=11)
            ax.set_ylabel('Parameter', fontsize=11)
            ax.grid(axis='x', alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes, fontsize=14)
            ax.axis('off')
    
    # Hide extra subplots if fewer than 4 algorithms
    for idx in range(len(algorithms), 4):
        axes[idx].axis('off')
    
    plt.suptitle(f'Hyperparameter Sensitivity by Algorithm ({metric.upper()})', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'parameter_sensitivity_by_algorithm_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: parameter_sensitivity_by_algorithm_{metric}.png")

def plot_sensitivity_heatmap(df, output_dir, metric='accuracy'):
    """
    Plot: Heatmap of Dataset × Algorithm sensitivity
    """
    # Filter by metric
    metric_df = df[df['metric'] == metric]
    
    # Create pivot table: dataset vs algorithm
    pivot = metric_df.pivot_table(values='variance', index='dataset', columns='algorithm', aggfunc='mean')
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.5f', cmap='YlOrRd', cbar_kws={'label': 'Average Variance'}, linewidths=0.5)
    plt.title(f'Sensitivity Heatmap: Dataset × Algorithm ({metric.upper()})', fontsize=16, fontweight='bold')
    plt.xlabel('Algorithm', fontsize=12)
    plt.ylabel('Dataset', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / f'sensitivity_heatmap_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: sensitivity_heatmap_{metric}.png")

def plot_parameter_comparison_across_datasets(df, output_dir, metric='accuracy'):
    """
    Plot: How does the same parameter behave across datasets?
    Focus on common parameters like C, k, max_depth, etc.
    """
    # Filter by metric
    metric_df = df[df['metric'] == metric]
    
    # Find most common parameters
    param_counts = metric_df.groupby('parameter').size().sort_values(ascending=False)
    common_params = param_counts.head(6).index.tolist()  # Top 6 parameters
    
    if len(common_params) == 0:
        print("No common parameters found")
        return
    
    # Create subplots
    n_params = len(common_params)
    fig, axes = plt.subplots((n_params + 1) // 2, 2, figsize=(16, 4 * ((n_params + 1) // 2)))
    
    if n_params == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, param in enumerate(common_params):
        ax = axes[idx]
        
        # Get data for this parameter
        param_data = metric_df[metric_df['parameter'] == param]
        
        # Plot variance by dataset
        pivot = param_data.pivot_table(values='variance', index='dataset', columns='algorithm', aggfunc='mean')
        
        if not pivot.empty:
            pivot.plot(kind='bar', ax=ax)
            ax.set_title(f'Parameter: {param}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dataset', fontsize=10)
            ax.set_ylabel('Variance', fontsize=10)
            ax.legend(title='Algorithm', fontsize=8, title_fontsize=9)
            ax.grid(axis='y', alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'No data for {param}', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
    
    # Hide extra subplots
    for idx in range(len(common_params), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Parameter Behavior Across Datasets ({metric.upper()})', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / f'parameter_comparison_across_datasets_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: parameter_comparison_across_datasets_{metric}.png")

def plot_metric_comparison(df, output_dir):
    """
    Plot: Compare sensitivity across different metrics (accuracy, f1, auc)
    """
    # Group by metric and calculate average variance
    metric_sens = df.groupby('metric')['variance'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    metric_sens.plot(kind='bar', color='skyblue')
    plt.title('Sensitivity by Evaluation Metric', fontsize=16, fontweight='bold')
    plt.xlabel('Metric', fontsize=12)
    plt.ylabel('Average Variance', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'metric_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: metric_comparison.png")

def plot_top_sensitive_combinations(df, output_dir, metric='accuracy', top_n=15):
    """
    Plot: Top N most sensitive dataset-algorithm-parameter combinations
    """
    # Filter by metric
    metric_df = df[df['metric'] == metric]
    
    # Create combination label
    metric_df['combination'] = (
        metric_df['dataset'] + ' + ' + 
        metric_df['algorithm'] + ' + ' + 
        metric_df['parameter']
    )
    
    # Sort by variance and take top N
    top_combos = metric_df.nlargest(top_n, 'variance')
    
    plt.figure(figsize=(14, 8))
    plt.barh(range(len(top_combos)), top_combos['variance'].values, color='coral')
    plt.yticks(range(len(top_combos)), top_combos['combination'].values)
    plt.xlabel('Variance', fontsize=12)
    plt.ylabel('Dataset + Algorithm + Parameter', fontsize=12)
    plt.title(f'Top {top_n} Most Sensitive Combinations ({metric.upper()})', fontsize=16, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / f'top_sensitive_combinations_{metric}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: top_sensitive_combinations_{metric}.png")

# ============================================================================
# STEP 4: GENERATE COMPREHENSIVE SUMMARY REPORT
# ============================================================================

def generate_comprehensive_report(df_raw, df_sensitivity, output_dir):
    """
    Generate detailed text summary of findings
    """
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE HYPERPARAMETER SENSITIVITY ANALYSIS")
    report.append("=" * 100)
    report.append("")
    
    # Overall statistics
    report.append("OVERALL STATISTICS")
    report.append("-" * 100)
    report.append(f"Total experiments analyzed: {len(df_raw)}")
    report.append(f"Number of datasets: {df_raw['dataset'].nunique()}")
    report.append(f"Number of algorithms: {df_raw['algorithm'].nunique()}")
    report.append(f"Number of parameters tested: {df_raw['param_name'].nunique()}")
    report.append("")
    
    # Dataset breakdown
    report.append("Datasets:")
    for dataset in sorted(df_raw['dataset'].unique()):
        count = len(df_raw[df_raw['dataset'] == dataset])
        report.append(f"  - {dataset}: {count} experiments")
    report.append("")
    
    # Algorithm breakdown
    report.append("Algorithms:")
    for algo in sorted(df_raw['algorithm'].unique()):
        count = len(df_raw[df_raw['algorithm'] == algo])
        report.append(f"  - {algo}: {count} experiments")
    report.append("")
    report.append("")
    
    # Analysis for each metric
    for metric in ['accuracy', 'f1', 'auc']:
        metric_df = df_sensitivity[df_sensitivity['metric'] == metric]
        
        if len(metric_df) == 0:
            continue
        
        report.append("=" * 100)
        report.append(f"ANALYSIS FOR {metric.upper()}")
        report.append("=" * 100)
        report.append("")
        
        # 1. Most/Least sensitive datasets
        dataset_sens = metric_df.groupby('dataset')['variance'].mean().sort_values(ascending=False)
        report.append(f"1. DATASET SENSITIVITY RANKING (by {metric.upper()})")
        report.append("-" * 100)
        for i, (dataset, var) in enumerate(dataset_sens.items(), 1):
            report.append(f"   {i:2d}. {dataset:20s} - Avg Variance: {var:.8f}")
        if len(dataset_sens) > 0:
            report.append(f"\n   Most Sensitive:  {dataset_sens.index[0]}")
            report.append(f"   Least Sensitive: {dataset_sens.index[-1]}")
        report.append("")
        
        # 2. Most/Least sensitive algorithms
        algo_sens = metric_df.groupby('algorithm')['variance'].mean().sort_values(ascending=False)
        report.append(f"2. ALGORITHM SENSITIVITY RANKING (by {metric.upper()})")
        report.append("-" * 100)
        for i, (algo, var) in enumerate(algo_sens.items(), 1):
            report.append(f"   {i:2d}. {algo:25s} - Avg Variance: {var:.8f}")
        if len(algo_sens) > 0:
            report.append(f"\n   Most Sensitive:  {algo_sens.index[0]}")
            report.append(f"   Least Sensitive: {algo_sens.index[-1]}")
        report.append("")
        
        # 3. Most important parameters per algorithm
        report.append(f"3. MOST IMPORTANT HYPERPARAMETERS PER ALGORITHM (by {metric.upper()})")
        report.append("-" * 100)
        for algorithm in sorted(metric_df['algorithm'].unique()):
            algo_data = metric_df[metric_df['algorithm'] == algorithm]
            param_sens = algo_data.groupby('parameter')['variance'].mean().sort_values(ascending=False)
            
            report.append(f"\n   {algorithm.upper().replace('_', ' ')}:")
            for i, (param, var) in enumerate(param_sens.head(5).items(), 1):  # Top 5
                report.append(f"      {i}. {param:20s} - Variance: {var:.8f}")
        report.append("")
        
        # 4. Top 10 sensitive combinations
        report.append(f"4. TOP 10 MOST SENSITIVE COMBINATIONS (by {metric.upper()})")
        report.append("-" * 100)
        top_combos = metric_df.nlargest(10, 'variance')
        for i, row in enumerate(top_combos.itertuples(), 1):
            combo = f"{row.dataset} + {row.algorithm} + {row.parameter}"
            report.append(f"   {i:2d}. {combo:60s} Var={row.variance:.8f} (Range: {row.range:.4f})")
        report.append("")
        report.append("")
    
    # Key insights section
    report.append("=" * 100)
    report.append("KEY INSIGHTS")
    report.append("=" * 100)
    report.append("")
    
    # Overall most important parameters
    report.append("1. UNIVERSALLY IMPORTANT PARAMETERS")
    report.append("-" * 100)
    universal_params = df_sensitivity.groupby('parameter')['variance'].mean().sort_values(ascending=False).head(10)
    for i, (param, var) in enumerate(universal_params.items(), 1):
        report.append(f"   {i:2d}. {param:20s} - Avg variance across all contexts: {var:.8f}")
    report.append("")
    
    # Dataset-specific insights
    report.append("2. DATASET-SPECIFIC INSIGHTS")
    report.append("-" * 100)
    for dataset in sorted(df_sensitivity['dataset'].unique()):
        dataset_data = df_sensitivity[df_sensitivity['dataset'] == dataset]
        
        # Most sensitive parameter for this dataset
        top_param = dataset_data.groupby('parameter')['variance'].mean().idxmax()
        top_var = dataset_data.groupby('parameter')['variance'].mean().max()
        
        # Best performing algorithm
        best_algo = df_raw[df_raw['dataset'] == dataset].groupby('algorithm')['accuracy'].mean().idxmax()
        best_acc = df_raw[df_raw['dataset'] == dataset].groupby('algorithm')['accuracy'].mean().max()
        
        report.append(f"\n   {dataset.upper()}:")
        report.append(f"      - Most sensitive to: '{top_param}' (variance={top_var:.8f})")
        report.append(f"      - Best algorithm: {best_algo} (avg accuracy={best_acc:.4f})")
    
    report.append("")
    report.append("")
    
    # Algorithm-specific insights
    report.append("3. ALGORITHM-SPECIFIC INSIGHTS")
    report.append("-" * 100)
    for algorithm in sorted(df_sensitivity['algorithm'].unique()):
        algo_data = df_sensitivity[df_sensitivity['algorithm'] == algorithm]
        
        # Most sensitive parameter
        top_param = algo_data.groupby('parameter')['variance'].mean().idxmax()
        top_var = algo_data.groupby('parameter')['variance'].mean().max()
        
        # Best performing dataset
        best_dataset = df_raw[df_raw['algorithm'] == algorithm].groupby('dataset')['accuracy'].mean().idxmax()
        best_acc = df_raw[df_raw['algorithm'] == algorithm].groupby('dataset')['accuracy'].mean().max()
        
        report.append(f"\n   {algorithm.upper().replace('_', ' ')}:")
        report.append(f"      - Most sensitive to: '{top_param}' (variance={top_var:.8f})")
        report.append(f"      - Works best on: {best_dataset} (avg accuracy={best_acc:.4f})")
    
    report.append("")
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)
    
    # Write to file
    with open(output_dir / 'comprehensive_analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print(f"✓ Saved: comprehensive_analysis_report.txt")
    
    # Also print summary to console
    print("\n" + "=" * 100)
    print("REPORT SUMMARY")
    print("=" * 100)
    for line in report[:30]:  # Print first 30 lines
        print(line)
    print("...")
    print(f"\nFull report saved to: {output_dir / 'comprehensive_analysis_report.txt'}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Configuration
    BASE_PATH = Path(__file__).resolve().parent.parent
    OUTPUT_DIR = BASE_PATH / 'analysis' / 'results'
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    print("=" * 100)
    print("COMPREHENSIVE HYPERPARAMETER SENSITIVITY ANALYSIS")
    print("Analyzing standardized JSON results from all datasets and algorithms")
    print("=" * 100)
    print()
    
    # Step 1: Find all JSON result files
    print("Step 1: Finding JSON result files...")
    result_files = find_json_result_files(BASE_PATH)
    print(f"Found {len(result_files)} result files:")
    for file in result_files:
        print(f"  - {file.parent.parent.name}/{file.parent.name}/{file.name}")
    print()
    
    # Step 2: Load all JSON files
    print("Step 2: Loading and parsing JSON files...")
    all_data = []
    for file_path in result_files:
        experiments = load_json_results(file_path)
        all_data.extend(experiments)
        print(f"  ✓ Loaded {len(experiments)} experiments from {file_path.name}")
    
    # Create DataFrame
    df_raw = pd.DataFrame(all_data)
    print(f"\nTotal experiments loaded: {len(df_raw)}")
    print()
    
    # Save raw data
    df_raw.to_csv(OUTPUT_DIR / 'all_experiments_raw.csv', index=False)
    print(f"✓ Saved: all_experiments_raw.csv")
    print()
    
    # Step 3: Calculate sensitivity metrics
    print("Step 3: Calculating sensitivity metrics...")
    df_sensitivity = calculate_sensitivity_metrics(df_raw)
    
    # Save sensitivity data
    df_sensitivity.to_csv(OUTPUT_DIR / 'sensitivity_metrics.csv', index=False)
    print(f"✓ Saved: sensitivity_metrics.csv")
    print(f"  Total sensitivity records: {len(df_sensitivity)}")
    print()
    
    # Step 4: Create visualizations
    print("Step 4: Creating visualizations...")
    print()
    
    # Create plots for each main metric
    for metric in ['accuracy', 'f1', 'auc']:
        print(f"  Creating plots for {metric.upper()}...")
        plot_dataset_sensitivity(df_sensitivity, OUTPUT_DIR, metric)
        plot_algorithm_sensitivity(df_sensitivity, OUTPUT_DIR, metric)
        plot_parameter_sensitivity_by_algorithm(df_sensitivity, OUTPUT_DIR, metric)
        plot_sensitivity_heatmap(df_sensitivity, OUTPUT_DIR, metric)
        plot_parameter_comparison_across_datasets(df_sensitivity, OUTPUT_DIR, metric)
        plot_top_sensitive_combinations(df_sensitivity, OUTPUT_DIR, metric)
        print()
    
    # Create metric comparison plot
    plot_metric_comparison(df_sensitivity, OUTPUT_DIR)
    print()
    
    # Step 5: Generate comprehensive report
    print("Step 5: Generating comprehensive report...")
    generate_comprehensive_report(df_raw, df_sensitivity, OUTPUT_DIR)
    print()
    
    print("=" * 100)
    print("ANALYSIS COMPLETE!")
    print(f"All results saved to: {OUTPUT_DIR}")
    print()
    print("Generated files:")
    print("  - all_experiments_raw.csv")
    print("  - sensitivity_metrics.csv")
    print("  - comprehensive_analysis_report.txt")
    print("  - Multiple visualization plots (PNG files)")
    print("=" * 100)

if __name__ == "__main__":
    main()
