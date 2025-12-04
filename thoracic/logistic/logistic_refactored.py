import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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

def load_and_preprocess(filepath, target_col):
    """Load data and perform preprocessing including label encoding for multiclass target."""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"Shape after dropping NaN: {df.shape}")
    else:
        print("No missing values found")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
    
    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])
        print(f"Encoded feature: {col}")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    Y = df[target_col]
    
    # Encode multiclass target variable
    Y = le.fit_transform(Y)
    
    # Check class balance
    print(f"\nClass distribution (Encoded: 0 to {np.max(Y)}):")
    unique, counts = np.unique(Y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y)*100:.2f}%)")
    
    return X, Y

def split_and_scale(X, Y):
    """Split data and scale features."""
    # Use stratify for fair representation of all classes in train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics with fix for Binary AUC"""
    # For multi-class, we use 'macro' or 'weighted' averaging. 
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    }
    
    if Y_proba is not None:
        try:
            # CHECK: If binary classification (2 columns), only take the 2nd column (Positive class)
            if Y_proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
            else:
                # For multi-class (>2 columns), use One-vs-Rest strategy
                metrics['auc'] = roc_auc_score(Y_true, Y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            metrics['auc'] = 0.0
            print(f"AUC calculation failed: {e}")
    else:
        metrics['auc'] = 0.0

    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger, C_values=[0.001, 0.01, 0.1, 1, 10, 100]):
    """
    Test sensitivity to regularization strength C.
    Uses 'lbfgs' solver and 'multinomial' scheme for L2 regularization.
    """
    results_list = []
    
    for C in C_values:
        # 'lbfgs' is efficient for L2 and multinomial.
        model = LogisticRegression(C=C, max_iter=5000, multi_class='multinomial', solver='lbfgs', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='C',
            param_value=C,
            metrics=test_metrics
        )
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # Plot multiple metrics
    plt.figure(figsize=(12, 6))
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(C_values, values, marker="o", label=metric.upper())
    
    plt.xscale("log")
    plt.xlabel("Regularization Strength C (log scale)")
    plt.ylabel("Test Score (Weighted)")
    plt.title("Sensitivity to C (Multiclass) - Thoracic")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_penalty_sensitivity(X_train, X_test, Y_train, Y_test, logger, C_fixed=1.0):
    """
    Test L1 vs L2 penalty with fixed C.
    Must use 'liblinear' solver and 'ovr' scheme for L1 multiclass support.
    """
    penalties = ['l1', 'l2']
    results_list = []
    
    for penalty in penalties:
        # 'liblinear' supports L1, but only with multi_class='ovr' for multiclass
        model = LogisticRegression(penalty=penalty, C=C_fixed, max_iter=5000, solver='liblinear', 
                                   multi_class='ovr', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='penalty',
            param_value=penalty,
            metrics=test_metrics,
            additional_info={'solver': 'liblinear'}
        )
        
        print(f"{penalty.upper()}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # Grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    l1_values = [results_list[0][m] for m in metrics_to_plot]
    l2_values = [results_list[1][m] for m in metrics_to_plot]
    
    plt.bar(x, l1_values, width, label='L1 (Feature Selection)', color='lightcoral')
    plt.bar(x + width, l2_values, width, label='L2 (Weight Shrinkage)', color='skyblue')
    
    plt.xlabel('Metrics')
    plt.ylabel('Test Score (Weighted)')
    plt.title(f'L1 vs L2 Penalty Comparison (C={C_fixed}) - Thoracic')
    plt.xticks(x + width / 2, metrics_to_plot)
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
        dataset="thoracic",
        algorithm="logistic_regression"
    )
    
    # load and preprocess
    X, Y = load_and_preprocess("../ThoraricSurgery.csv", "Risk1Yr")
    
    # split and scale (CRITICAL for Regularized Logistic Regression!)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*70)
    print("Running C (Regularization) sensitivity test (Solver: lbfgs, Scheme: multinomial)")
    print("="*70)
    C_results = test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*70)
    print("Running L1 vs L2 penalty comparison (Solver: liblinear, Scheme: ovr)")
    print("="*70)
    penalty_results = test_penalty_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: logistic_regression_sensitivity_results.json")
    print("  - Text file: logistic_regression_sensitivity_results.txt")
    print("  - Plots: lr_c_sensitivity_plot.png, lr_penalty_comparison_plot.png")
    print("="*80)
