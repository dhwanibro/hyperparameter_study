import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings

# Suppress convergence warnings to keep output clean (since we handled them with max_iter)
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
    """Helper to run the 5-fold loop inside your test functions"""
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    fold_stats = {'test': [], 'train': [], 'n_support': []}
    
    for train_ix, test_ix in skf.split(X, Y):
        # Split
        X_train_fold, X_test_fold = X.iloc[train_ix], X.iloc[test_ix]
        Y_train_fold, Y_test_fold = Y.iloc[train_ix], Y.iloc[test_ix]
        
        # Scale (Fit on train, transform test)
        X_train_fold = scaler.fit_transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # Fit Model with Safety Limit
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
        # Calculate variance across folds (stability)
        avg_results['variances'][metric] = np.var([x[metric] for x in fold_stats['test']])
        
    avg_results['n_support'] = int(np.mean(fold_stats['n_support']))
    
    return avg_results

def test_C_sensitivity(X, Y, C_values=[0.01, 0.1, 1, 10, 100]):
    """Test sensitivity to C hyperparameter using 5-Fold CV"""
    results_list = []
    C_results = {}
    training_times = {}
    
    print(f"\nStarting C Sensitivity Analysis ({CV_FOLDS}-Fold CV)...")
    
    for C in C_values:
        print(f"Testing C={C}...", end="", flush=True)
        start_time = time.time()
        
        # Run 5-Fold CV
        # Using linear kernel for C-test to keep it fast
        params = {'C': C, 'kernel': 'linear'}
        metrics = run_cv_fold(X, Y, params)
        
        training_time = (time.time() - start_time) / CV_FOLDS # Average time per fold
        training_times[C] = training_time
        
        metrics['training_time'] = training_time
        results_list.append(metrics['test'])
        C_results[C] = metrics
        
        print(f" Done. Avg Acc={metrics['test']['accuracy']:.4f}")
    
    # find best performing C
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_val = max(C_results.items(), key=lambda x: x[1]['test'][metric])
        best_by_metric[metric] = best_val[0]
    
    # === PLOTTING (Restored your original plotting style) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance metrics
    for metric in ['accuracy', 'f1', 'auc']:
        values = [C_results[c]['test'][metric] for c in C_values]
        ax1.plot(range(len(C_values)), values, marker="o", label=metric.upper())
    
    ax1.set_xticks(range(len(C_values)))
    ax1.set_xticklabels([str(c) for c in C_values])
    ax1.set_xlabel("C (Regularization Parameter)")
    ax1.set_ylabel("5-Fold CV Score")
    ax1.set_title("Sensitivity to C (CV Average)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log') # kept log scale if using 0.01 etc
    
    # Plot 2: Number of support vectors
    support_vectors = [C_results[c]['n_support'] for c in C_values]
    ax2.plot(range(len(C_values)), support_vectors, marker="s", color='red', linewidth=2)
    ax2.set_xticks(range(len(C_values)))
    ax2.set_xticklabels([str(c) for c in C_values])
    ax2.set_xlabel("C (Regularization Parameter)")
    ax2.set_ylabel("Avg Support Vectors")
    ax2.set_title("Model Complexity vs C")
    ax2.grid(True)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig("svm_C_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return C_results, best_by_metric, training_times

def test_kernel_sensitivity(X, Y):
    """Test sensitivity to kernel type using 5-Fold CV"""
    # Removed 'poly' and 'sigmoid' for safety on this dataset, kept reliable ones
    kernels = ['linear', 'rbf'] 
    results = {}
    training_times = {}
    
    print(f"\nStarting Kernel Comparison ({CV_FOLDS}-Fold CV)...")
    
    for kern in kernels:
        print(f"Testing kernel={kern}...", end="", flush=True)
        start_time = time.time()
        
        params = {'kernel': kern, 'C': 1.0}
        metrics = run_cv_fold(X, Y, params)
        
        training_time = (time.time() - start_time) / CV_FOLDS
        training_times[kern] = training_time
        
        metrics['training_time'] = training_time
        results[kern] = metrics
        
        print(f" Done. Avg Acc={metrics['test']['accuracy']:.4f}")
    
    # find best performing kernel
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_val = max(results.items(), key=lambda x: x[1]['test'][metric])
        best_by_metric[metric] = best_val[0]
    
    # === PLOTTING (Restored your original plotting style) ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, kern in enumerate(kernels):
        values = [results[kern]['test'][m] for m in metrics_to_plot]
        ax1.bar(x + i*width, values, width, label=kern)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('CV Score')
    ax1.set_title('Kernel Comparison (All Metrics)')
    ax1.set_xticks(x + width * 0.5)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Training time comparison
    times = [training_times.get(k, 0) for k in kernels]
    ax2.bar(kernels, times, color=['blue', 'green'])
    ax2.set_xlabel('Kernel Type')
    ax2.set_ylabel('Avg Training Time (seconds)')
    ax2.set_title('Training Time by Kernel')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("svm_kernel_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, best_by_metric, training_times

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess (Includes Downsampling now)
    X, Y = load_and_preprocess("../diabetes_binary.csv", "Diabetes_binary")
    
    # Note: We do NOT scale here anymore. Scaling happens inside the Cross Validation 
    # loops to prevent data leakage, which is best practice for CV.
    
    # run tests
    print("\n" + "="*60)
    print("Running C (regularization) sensitivity test...")
    print("="*60)
    C_results, best_C, C_times = test_C_sensitivity(X, Y)
    
    print("\n" + "="*60)
    print("Running kernel comparison...")
    print("="*60)
    kernel_results, best_kernel, kernel_times = test_kernel_sensitivity(X, Y)
    
    # === WRITING THE DETAILED REPORT (Restored your original text format) ===
    with open("svm_sensitivity_results.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("SVM HYPERPARAMETER SENSITIVITY ANALYSIS - DIABETES DATASET\n")
        f.write(f"Technique: {CV_FOLDS}-Fold Cross Validation\n")
        f.write(f"Sample Size: {SAMPLE_SIZE} (Stratified)\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: C parameter
        f.write("TEST 1: C (REGULARIZATION PARAMETER) SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual C results (Averages across 5 folds):\n")
        for C, metrics in C_results.items():
            test = metrics['test']
            train = metrics['train']
            var = metrics['variances']
            
            f.write(f"C = {C}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
            f.write(f"  Train Metrics: Acc={train['accuracy']:.4f}, F1={train['f1']:.4f}\n")
            f.write(f"  Avg Support Vectors: {metrics['n_support']}\n")
            f.write(f"  Avg Training Time: {metrics['training_time']:.2f}s\n")
            f.write(f"  Overfitting Gap (Train-Test Acc): {train['accuracy']-test['accuracy']:.4f}\n")
            f.write(f"  Stability (Test Acc Variance): {var['accuracy']:.6f}\n\n")
        
        f.write(f"\nBest C by metric:\n")
        for metric, C in best_C.items():
            f.write(f"  {metric}: {C}\n")
        f.write(f"\nPlot saved as: svm_C_sensitivity_plot.png\n\n")
        
        f.write("WHY: C controls the regularization strength.\n")
        f.write("With 5-Fold CV, we can see if higher C leads to overfitting (High Train Acc vs Low Test Acc).\n\n")
        
        # TEST 2: kernel
        f.write("TEST 2: KERNEL TYPE COMPARISON\n")
        f.write("-"*80 + "\n\n")
        for kern, metrics in kernel_results.items():
            test = metrics['test']
            train = metrics['train']
            f.write(f"{kern}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
            f.write(f"  Train Metrics: Acc={train['accuracy']:.4f}, F1={train['f1']:.4f}\n")
            f.write(f"  Avg Support Vectors: {metrics['n_support']}\n")
            f.write(f"  Avg Training Time: {metrics['training_time']:.2f}s\n\n")
        
        f.write(f"\nBest kernel by metric:\n")
        for metric, kern in best_kernel.items():
            f.write(f"  {metric}: {kern}\n")
        f.write(f"\nPlot saved as: svm_kernel_comparison_plot.png\n\n")
        
        # SUMMARY
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"\nBest hyperparameters by metric:\n")
        f.write(f"\n  For ACCURACY:\n")
        f.write(f"    C: {best_C['accuracy']}\n")
        f.write(f"    kernel: {best_kernel['accuracy']}\n")
        
        f.write(f"\n  For F1:\n")
        f.write(f"    C: {best_C['f1']}\n")
        f.write(f"    kernel: {best_kernel['f1']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY OBSERVATIONS:\n")
        f.write("1. If C is high and Overfitting Gap is large, reduce C.\n")
        f.write("2. 'max_iter' was used to prevent the SVM from hanging on non-converging folds.\n")
        f.write("3. If Linear and RBF have similar accuracy, Linear is preferred for speed.\n")
        
    print("\n" + "="*60)
    print("Results saved to: svm_sensitivity_results.txt")
    print("Plots saved as: svm_C_sensitivity_plot.png, svm_kernel_comparison_plot.png")
    print("="*60)