import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

# ### The target variable is "NObeyesdad" (obesity level)
# Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III

def load_and_preprocess(filepath, target_col):
    """Load data and do preprocessing"""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"Shape after dropping NaN: {df.shape}")
    else:
        print("No missing values found")
    
    # check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
    
    # encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded {col}")
    
    # separate features and target
    X = df.drop(target_col, axis=1)
    Y = df[target_col]
    
    # encode target variable
    Y = le.fit_transform(Y)
    
    # check class balance
    print(f"\nClass distribution:")
    unique, counts = np.unique(Y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y)*100:.2f}%)")
    
    return X, Y

def split_and_scale(X, Y):
    """Split data and scale features"""
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics"""
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    }
    if Y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(Y_true, Y_proba, multi_class='ovr', average='weighted')
        except:
            metrics['auc'] = 0.0
    return metrics

def test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test, logger, depth_values=[1, 2, 3, 5, 7, 10, 15, 20, None]):
    """Test sensitivity to max_depth hyperparameter"""
    results_list = []
    
    for depth in depth_values:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='max_depth',
            param_value=str(depth) if depth is not None else 'None',
            metrics=test_metrics
        )
        
        print(f"Depth={depth}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    x_labels = [str(d) for d in depth_values]
    x_positions = range(len(depth_values))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(x_positions, values, marker="o", label=metric.upper())
    
    plt.xticks(x_positions, x_labels)
    plt.xlabel("max_depth")
    plt.ylabel("Score")
    plt.title("Sensitivity to max_depth (Multiple Metrics) - Obesity")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_depth_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test, logger, split_values=[2, 5, 10, 20, 50, 100]):
    """Test sensitivity to min_samples_split hyperparameter"""
    results_list = []
    
    for split in split_values:
        model = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='min_samples_split',
            param_value=split,
            metrics=test_metrics
        )
        
        print(f"split={split}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(split_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    plt.title("Sensitivity to min_samples_split (Multiple Metrics) - Obesity")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_split_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test, logger, leaf_values=[1, 2, 5, 10, 20, 50]):
    """Test sensitivity to min_samples_leaf hyperparameter"""
    results_list = []
    
    for leaf in leaf_values:
        model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='min_samples_leaf',
            param_value=leaf,
            metrics=test_metrics
        )
        
        print(f"leaf={leaf}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(leaf_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Score")
    plt.title("Sensitivity to min_samples_leaf (Multiple Metrics) - Obesity")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_leaf_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_criterion_sensitivity(X_train, X_test, Y_train, Y_test, logger):
    """Test sensitivity to criterion (splitting criterion)"""
    criteria = ['gini', 'entropy', 'log_loss']
    results_list = []
    
    for crit in criteria:
        model = DecisionTreeClassifier(criterion=crit, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='criterion',
            param_value=crit,
            metrics=test_metrics
        )
        
        print(f"{crit}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, crit in enumerate(criteria):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=crit)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Criterion Comparison (All Metrics) - Obesity')
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("dt_criterion_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="obesity",
        algorithm="decision_tree"
    )
    
    # load and preprocess
    X, Y = load_and_preprocess("../obesity.csv", "NObeyesdad")
    
    # split and scale
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*60)
    print("Running max_depth sensitivity test...")
    print("="*60)
    depth_results = test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running min_samples_split sensitivity test...")
    print("="*60)
    split_results = test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running min_samples_leaf sensitivity test...")
    print("="*60)
    leaf_results = test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running criterion comparison...")
    print("="*60)
    criterion_results = test_criterion_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: decision_tree_sensitivity_results.json")
    print("  - Text file: decision_tree_sensitivity_results.txt")
    print("  - Plots: dt_depth_sensitivity_plot.png, dt_split_sensitivity_plot.png,")
    print("           dt_leaf_sensitivity_plot.png, dt_criterion_comparison_plot.png")
    print("="*80)
