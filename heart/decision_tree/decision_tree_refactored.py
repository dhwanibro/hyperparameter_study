import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

# Add analysis directory to path
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

warnings.filterwarnings("ignore")

def load_and_preprocess(filepath, target_col):
    """Load data and preprocess"""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"Shape after dropping NaN: {df.shape}")
    else:
        print("No missing values found")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Convert {1,2} → {0,1}
    y = y.replace({1: 0, 2: 1})
    
    print(f"\nClass distribution:\n{y.value_counts()}")
    print(f"Class proportions:\n{y.value_counts(normalize=True)}")
    
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split and scale the data"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def calculate_all_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def test_max_depth_sensitivity(X_train, X_test, y_train, y_test, logger, depth_values=list(range(1, 21))):
    """Test sensitivity to max_depth hyperparameter"""
    results_list = []
    
    for depth in depth_values:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='max_depth',
            param_value=depth,
            metrics=metrics
        )
        
        print(f"max_depth={depth}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(depth_values, [r['accuracy'] for r in results_list], marker='o', label='Accuracy')
    plt.plot(depth_values, [r['f1'] for r in results_list], marker='s', label='F1')
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.title("Decision Tree Sensitivity to Max Depth - Heart Disease")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_depth_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_min_samples_split_sensitivity(X_train, X_test, y_train, y_test, logger, split_values=list(range(2, 21))):
    """Test sensitivity to min_samples_split hyperparameter"""
    results_list = []
    
    for split in split_values:
        clf = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='min_samples_split',
            param_value=split,
            metrics=metrics
        )
        
        print(f"min_samples_split={split}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(split_values, [r['accuracy'] for r in results_list], marker='o', label='Accuracy')
    plt.plot(split_values, [r['f1'] for r in results_list], marker='s', label='F1')
    plt.xlabel("Min Samples Split")
    plt.ylabel("Score")
    plt.title("Decision Tree Sensitivity to Min Samples Split - Heart Disease")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_split_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_min_samples_leaf_sensitivity(X_train, X_test, y_train, y_test, logger, leaf_values=list(range(1, 21))):
    """Test sensitivity to min_samples_leaf hyperparameter"""
    results_list = []
    
    for leaf in leaf_values:
        clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='min_samples_leaf',
            param_value=leaf,
            metrics=metrics
        )
        
        print(f"min_samples_leaf={leaf}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_values, [r['accuracy'] for r in results_list], marker='o', label='Accuracy')
    plt.plot(leaf_values, [r['f1'] for r in results_list], marker='s', label='F1')
    plt.xlabel("Min Samples Leaf")
    plt.ylabel("Score")
    plt.title("Decision Tree Sensitivity to Min Samples Leaf - Heart Disease")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_leaf_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_criterion_comparison(X_train, X_test, y_train, y_test, logger):
    """Test sensitivity to criterion parameter"""
    criteria = ["gini", "entropy"]
    results_list = []
    
    for criterion in criteria:
        clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='criterion',
            param_value=criterion,
            metrics=metrics
        )
        
        print(f"criterion={criterion}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, criterion in enumerate(criteria):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=criterion)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Decision Tree Criterion Comparison - Heart Disease')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("dt_criterion_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def main():
    # Initialize logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="heart",
        algorithm="decision_tree"
    )
    
    # Load and preprocess
    X, y = load_and_preprocess("../heart.csv", "target")
    
    # Split and scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    # Run tests
    print("\n" + "="*60)
    print("Running max_depth sensitivity test...")
    print("="*60)
    depth_results = test_max_depth_sensitivity(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60)
    print("Running min_samples_split sensitivity test...")
    print("="*60)
    split_results = test_min_samples_split_sensitivity(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60)
    print("Running min_samples_leaf sensitivity test...")
    print("="*60)
    leaf_results = test_min_samples_leaf_sensitivity(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60)
    print("Running criterion comparison...")
    print("="*60)
    criterion_results = test_criterion_comparison(X_train, X_test, y_train, y_test, logger)
    
    # Save all results
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: decision_tree_sensitivity_results.json")
    print("  - Text file: decision_tree_sensitivity_results.txt")
    print("  - Plots: dt_depth_sensitivity_plot.png, dt_split_sensitivity_plot.png, dt_leaf_sensitivity_plot.png, dt_criterion_comparison_plot.png")
    print("="*80)

if __name__ == "__main__":
    main()
