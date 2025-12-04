import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

# Suppress convergence warnings
warnings.filterwarnings("ignore")

# ### Target variable: Severity (0 = benign, 1 = malignant)

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
    
    # separate features and target
    X = df.drop(target_col, axis=1)
    Y = df[target_col]
    
    # check class balance
    print(f"\nClass distribution:")
    print(Y.value_counts())
    print(f"Class balance: {Y.value_counts(normalize=True)}")
    
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
        'precision': precision_score(Y_true, Y_pred, zero_division=0),
        'recall': recall_score(Y_true, Y_pred, zero_division=0),
        'f1': f1_score(Y_true, Y_pred, zero_division=0)
    }
    if Y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(Y_true, Y_proba)
        except:
            metrics['auc'] = 0.0
    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger, C_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    """Test sensitivity to C hyperparameter (regularization strength)"""
    results_list = []
    
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=10000, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)[:, 1]
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='C',
            param_value=C,
            metrics=test_metrics
        )
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(range(len(C_values)), values, marker="o", label=metric.upper())
    
    plt.xticks(range(len(C_values)), [str(c) for c in C_values])
    plt.xlabel("C (Inverse Regularization Strength)")
    plt.ylabel("Score")
    plt.title("Sensitivity to C (Multiple Metrics) - Mammogram")
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.savefig("lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_penalty_sensitivity(X_train, X_test, Y_train, Y_test, logger):
    """Test sensitivity to penalty type"""
    penalties = ['l1', 'l2']
    solvers = ['liblinear', 'lbfgs']  # l1 needs liblinear, l2 can use lbfgs
    results_list = []
    
    for penalty, solver in zip(penalties, solvers):
        model = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)[:, 1]
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='penalty',
            param_value=penalty,
            metrics=test_metrics,
            additional_info={'solver': solver}
        )
        
        print(f"{penalty} ({solver}): Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, penalty in enumerate(penalties):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=penalty)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Penalty Comparison (All Metrics) - Mammogram')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="mammogram",
        algorithm="logistic_regression"
    )
    
    # load and preprocess
    X, Y = load_and_preprocess("../mamogram.csv", "Severity")
    
    # split and scale
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*60)
    print("Running C (regularization) sensitivity test...")
    print("="*60)
    C_results = test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running penalty comparison...")
    print("="*60)
    penalty_results = test_penalty_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: logistic_regression_sensitivity_results.json")
    print("  - Text file: logistic_regression_sensitivity_results.txt")
    print("  - Plots: lr_c_sensitivity_plot.png, lr_penalty_comparison_plot.png")
    print("="*80)
