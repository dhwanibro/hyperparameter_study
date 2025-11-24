import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ### The target variable is "NObeyesdad" (obesity level)
# This is a MULTICLASS Classification problem (7 classes).

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
    # Use stratify for fair representation of all 7 classes in train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def calculate_all_metrics_multiclass(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics for MULTICLASS classification using weighted average."""
    # Weighted averaging accounts for class imbalance
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    }
    
    if Y_proba is not None and Y_proba.shape[1] > 1:
        # Use OVR (One-vs-Rest) and weighted averaging for multiclass AUC
        try:
            metrics['auc'] = roc_auc_score(Y_true, Y_proba, multi_class='ovr', average='weighted')
        except Exception:
            metrics['auc'] = 0.0
    else:
        metrics['auc'] = 0.0
    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, C_values=[0.001, 0.01, 0.1, 1, 10, 100]):
    """
    Test sensitivity to regularization strength C.
    Uses 'lbfgs' solver and 'multinomial' scheme for L2 regularization.
    """
    results_list = []
    c_results = {}
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for C in C_values:
        # 'lbfgs' is efficient for L2 and multinomial.
        model = LogisticRegression(C=C, max_iter=5000, multi_class='multinomial', solver='lbfgs', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics_multiclass(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        c_results[C] = {'test': test_metrics}
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # Calculate variance for each metric
    metric_variances = {}
    for metric in metrics_to_check:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # Find best performing C for different metrics
    best_by_metric = {}
    for metric in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = C_values[best_idx]
    
    # Plot multiple metrics
    plt.figure(figsize=(12, 6))
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(C_values, values, marker="o", label=metric.upper())
    
    plt.xscale("log")
    plt.xlabel("Regularization Strength C (log scale)")
    plt.ylabel("Test Score (Weighted)")
    plt.title("Sensitivity to C (Multiclass) - Obesity")
    plt.legend()
    plt.grid(True)
    plt.savefig("obesity_lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return c_results, metric_variances, best_by_metric

def test_penalty_types(X_train, X_test, Y_train, Y_test, C_fixed=1.0):
    """
    Test L1 vs L2 penalty with fixed C.
    Must use 'liblinear' solver and 'ovr' scheme for L1 multiclass support.
    """
    penalties = ['l1', 'l2']
    results = {}
    results_list = []
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for penalty in penalties:
        # 'liblinear' supports L1, but only with multi_class='ovr' for multiclass
        model = LogisticRegression(penalty=penalty, C=C_fixed, max_iter=5000, solver='liblinear', 
                                   multi_class='ovr', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics_multiclass(Y_test, Y_test_pred, Y_test_proba)
        
        results[penalty] = {'test': test_metrics}
        results_list.append(test_metrics)
        
        print(f"{penalty.upper()}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # Calculate variance (difference) for each metric
    metric_variances = {}
    for metric in metrics_to_check:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # Find best performing penalty type for different metrics
    best_by_metric = {}
    for metric in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = penalties[best_idx]
    
    # Grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    l1_values = [results['l1']['test'][m] for m in metrics_to_plot]
    l2_values = [results['l2']['test'][m] for m in metrics_to_plot]
    
    plt.bar(x, l1_values, width, label='L1 (Feature Selection)', color='lightcoral')
    plt.bar(x + width, l2_values, width, label='L2 (Weight Shrinkage)', color='skyblue')
    
    plt.xlabel('Metrics')
    plt.ylabel('Test Score (Weighted)')
    plt.title(f'L1 vs L2 Penalty Comparison (C={C_fixed}) - Obesity')
    plt.xticks(x + width / 2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("obesity_lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, metric_variances, best_by_metric


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess
    X, Y = load_and_preprocess("../obesity.csv", "NObeyesdad")
    
    # split and scale (CRITICAL for Regularized Logistic Regression!)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*70)
    print("Running C (Regularization) sensitivity test (Solver: lbfgs, Scheme: multinomial)")
    print("="*70)
    c_results, c_variances, best_c = test_C_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*70)
    print("Running L1 vs L2 penalty comparison (Solver: liblinear, Scheme: ovr)")
    print("="*70)
    penalty_results, penalty_variances, best_penalty = test_penalty_types(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("lr_sensitivity_results_obesity.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("LOGISTIC REGRESSION HYPERPARAMETER SENSITIVITY ANALYSIS - OBESITY DATASET (MULTICLASS)\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: C sensitivity
        f.write("TEST 1: C SENSITIVITY (Regularization Strength) - L2 Penalty (multinomial)\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        f.write("Individual C value results (Test Metrics):\n")
        for C, metrics in c_results.items():
            test = metrics['test']
            f.write(f"C = {C}:\n")
            f.write(f"  Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric (over different C values):\n")
        for metric, var in c_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest C by metric:\n")
        for metric, c_val in best_c.items():
            f.write(f"  {metric}: {c_val}\n")
        f.write(f"Plot saved as: obesity_lr_c_sensitivity_plot.png\n\n")
        
        # TEST 2: Penalty type
        f.write("TEST 2: L1 vs L2 PENALTY COMPARISON (C=1.0) - OVR Scheme\n")
        f.write("--------------------------------------------------------------------------------\n\n")
        for penalty, metrics in penalty_results.items():
            test = metrics['test']
            f.write(f"{penalty.upper()}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance (difference) for each metric between L1 and L2:\n")
        for metric, var in penalty_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest Penalty by metric:\n")
        for metric, penalty in best_penalty.items():
            f.write(f"  {metric}: {penalty.upper()}\n")
        f.write(f"Plot saved as: obesity_lr_penalty_comparison_plot.png\n\n")
        
        # SUMMARY
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # calculate average variance across all metrics for each parameter
        avg_variances = {
            'C_value': np.mean(list(c_variances.values())),
            'penalty_type': np.mean(list(penalty_variances.values()))
        }
        
        most_sensitive = max(avg_variances, key=avg_variances.get)
        least_sensitive = min(avg_variances, key=avg_variances.get)
        
        f.write(f"Most sensitive parameter: {most_sensitive} (avg variance = {avg_variances[most_sensitive]:.6f})\n")
        f.write(f"Least sensitive parameter: {least_sensitive} (avg variance = {avg_variances[least_sensitive]:.6f})\n\n")
        
        f.write("Sensitivity ranking (by average variance across all metrics):\n")
        for param, var in sorted(avg_variances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {param}: {var:.6f}\n")
        
        f.write(f"\nBest hyperparameters by metric:\n")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        for metric in metrics:
            f.write(f"\n  For {metric.upper()}:\n")
            f.write(f"    Best C: {best_c[metric]} (from L2/Multinomial test)\n")
            f.write(f"    Best Penalty: {best_penalty[metric].upper()} (from L1/L2 OVR test)\n")

    print("\nResults saved to: lr_sensitivity_results_obesity.txt")
    print("Plots saved as: obesity_lr_c_sensitivity_plot.png, obesity_lr_penalty_comparison_plot.png")