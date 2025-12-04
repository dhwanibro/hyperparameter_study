import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

# Suppress convergence warnings to keep output clean
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLE_SIZE = 5000   # Keeps it fast enough to finish
CV_FOLDS = 5         # 5-Fold Cross Validation
MAX_ITER = 10000     # Safety limit to prevent freezing
# ==========================================

def load_and_preprocess(filepath, target_col):
    """Load data, do preprocessing, and downsample for SVM safety"""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        df = df.dropna()
        print(f"Shape after dropping NaN: {df.shape}")
    
    # check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
    
    # === SAFETY FIX: DOWNSAMPLING ===
    if len(df) > SAMPLE_SIZE:
        print(f"\n[SAFETY FIX] Dataset is too large for SVM ({len(df)} rows).")
        print(f"Downsampling to {SAMPLE_SIZE} stratifed samples...")
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=SAMPLE_SIZE, random_state=42)
        for _, sample_index in splitter.split(df, df[target_col]):
            df = df.iloc[sample_index]
        print(f"Shape for Training: {df.shape}")
    
    # separate features and target
    X = df.drop(target_col, axis=1)
    Y = df[target_col]
    
    # check class balance
    print(f"\nClass distribution:")
    unique, counts = np.unique(Y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y)*100:.2f}%)")
    
    return X, Y

def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics for binary classification"""
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='binary', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='binary', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='binary', zero_division=0)
    }
    if Y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
        except:
            metrics['auc'] = 0.5
    else:
        metrics['auc'] = 0.5
    return metrics

def run_cv_fold(X, Y, model_params):
    """Helper to run the 5-fold loop"""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    fold_stats = {'test': [], 'train': [], 'n_support': []}
    
    for train_ix, test_ix in skf.split(X, Y):
        # Split
        X_train_fold, X_test_fold = X.iloc[train_ix], X.iloc[test_ix]
        Y_train_fold, Y_test_fold = Y.iloc[train_ix], Y.iloc[test_ix]
        
        # Scale
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Fit Model
        model = SVC(**model_params, max_iter=MAX_ITER, probability=True, random_state=42)
        model.fit(X_train_fold, Y_train_fold)
        
        # Predict
        Y_train_pred = model.predict(X_train_fold)
        Y_test_pred = model.predict(X_test_fold)
        Y_test_proba = model.predict_proba(X_test_fold)
        
        # Save metrics
        fold_stats['train'].append(calculate_all_metrics(Y_train_fold, Y_train_pred))
        fold_stats['test'].append(calculate_all_metrics(Y_test_fold, Y_test_pred, Y_test_proba))
        fold_stats['n_support'].append(np.sum(model.n_support_))

    # Average results across folds
    avg_results = {'train': {}, 'test': {}, 'n_support': 0, 'variances': {}}
    
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        avg_results['test'][metric] = np.mean([x[metric] for x in fold_stats['test']])
        avg_results['train'][metric] = np.mean([x[metric] for x in fold_stats['train']])
        avg_results['variances'][metric] = np.var([x[metric] for x in fold_stats['test']])
        
    avg_results['n_support'] = int(np.mean(fold_stats['n_support']))
    
    return avg_results

def test_C_sensitivity(X, Y, logger, C_values=[0.01, 0.1, 1, 10, 100]):
    """Test sensitivity to C hyperparameter using 5-Fold CV"""
    results_list = []
    
    print(f"\nStarting C Sensitivity Analysis ({CV_FOLDS}-Fold CV)...")
    
    for C in C_values:
        print(f"Testing C={C}...", end="", flush=True)
        start_time = time.time()
        
        # Run 5-Fold CV
        params = {'C': C, 'kernel': 'linear'}
        metrics = run_cv_fold(X, Y, params)
        
        training_time = (time.time() - start_time) / CV_FOLDS
        results_list.append(metrics['test'])
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='C',
            param_value=C,
            metrics=metrics['test'],
            additional_info={
                'n_support_vectors': metrics['n_support'],
                'training_time': training_time,
                'kernel': 'linear'
            }
        )
        
        print(f" Done. Avg Acc={metrics['test']['accuracy']:.4f}")
    
    # === PLOTTING ===
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Plot performance metrics
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        ax.plot(range(len(C_values)), values, marker="o", label=metric.upper())
    
    ax.set_xticks(range(len(C_values)))
    ax.set_xticklabels([str(c) for c in C_values])
    ax.set_xlabel("C (Regularization Parameter)")
    ax.set_ylabel("5-Fold CV Score")
    ax.set_title("Sensitivity to C (CV Average) - Mammogram")
    ax.legend()
    ax.grid(True)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig("svm_C_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_kernel_sensitivity(X, Y, logger):
    """Test sensitivity to kernel type using 5-Fold CV"""
    kernels = ['linear', 'rbf'] 
    results_list = []
    
    print(f"\nStarting Kernel Comparison ({CV_FOLDS}-Fold CV)...")
    
    for kern in kernels:
        print(f"Testing kernel={kern}...", end="", flush=True)
        start_time = time.time()
        
        params = {'kernel': kern, 'C': 1.0}
        metrics = run_cv_fold(X, Y, params)
        
        training_time = (time.time() - start_time) / CV_FOLDS
        results_list.append(metrics['test'])
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='kernel',
            param_value=kern,
            metrics=metrics['test'],
            additional_info={
                'n_support_vectors': metrics['n_support'],
                'training_time': training_time,
                'C': 1.0
            }
        )
        
        print(f" Done. Avg Acc={metrics['test']['accuracy']:.4f}")
    
    # === PLOTTING ===
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Performance comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, kern in enumerate(kernels):
        values = [results_list[i][m] for m in metrics_to_plot]
        ax.bar(x + i*width, values, width, label=kern)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('CV Score')
    ax.set_title('Kernel Comparison (All Metrics) - Mammogram')
    ax.set_xticks(x + width * 0.5)
    ax.set_xticklabels(metrics_to_plot)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("svm_kernel_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="mammogram",
        algorithm="svm"
    )
    
    # load and preprocess
    X, Y = load_and_preprocess("../mamogram.csv", "Severity")
    
    # run tests
    print("\n" + "="*60)
    print("Running C (regularization) sensitivity test...")
    print("="*60)
    C_results = test_C_sensitivity(X, Y, logger)
    
    print("\n" + "="*60)
    print("Running kernel comparison...")
    print("="*60)
    kernel_results = test_kernel_sensitivity(X, Y, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: svm_sensitivity_results.json")
    print("  - Text file: svm_sensitivity_results.txt")
    print("  - Plots: svm_C_sensitivity_plot.png, svm_kernel_comparison_plot.png")
    print("="*80)
