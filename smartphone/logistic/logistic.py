import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ### The target variable is "Activity" (6 classes)
# 1 = WALKING, 2 = WALKING_UPSTAIRS, 3 = WALKING_DOWNSTAIRS,
# 4 = SITTING, 5 = STANDING, 6 = LAYING

def load_and_preprocess(train_path, test_path, target_col):
    """Load data and perform preprocessing (missing values, duplicates)."""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Check and handle missing values and duplicates for both datasets
    for name, df in [('train', train_df), ('test', test_df)]:
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"Missing values in {name}:\n{missing[missing > 0]}")
            df.dropna(inplace=True)
            print(f"Shape after dropping NaN in {name}: {df.shape}")
        else:
            print(f"No missing values in {name}")
            
        # Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows in {name}, removing...")
            df.drop_duplicates(inplace=True)
            print(f"Shape after dropping duplicates in {name}: {df.shape}")

    # Separate features and target
    X_train = train_df.drop(target_col, axis=1)
    Y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    Y_test = test_df[target_col]
    label_encoder = LabelEncoder()
    Y_train = label_encoder.fit_transform(Y_train)
    Y_test = label_encoder.transform(Y_test)
    
    print(f"\nLabel encoding mapping:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"{i}: {label}")
    
    # Check class balance
    print(f"\nTrain class distribution:")
    unique, counts = np.unique(Y_train, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y_train)*100:.2f}%)")
    
    print(f"\nTest class distribution:")
    unique, counts = np.unique(Y_test, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y_test)*100:.2f}%)")
    
    return X_train, X_test, Y_train, Y_test

def scale_features(X_train, X_test):
    """Scale features using StandardScaler"""
    print(f"\nScaling features...")
    scaler = StandardScaler()
    
    # Fit scaler on training data only to prevent data leakage
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

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
    Test sensitivity to regularization strength C using L2 penalty (multinomial).
    """
    results_list = []
    c_results = {}
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for C in C_values:
        # 'lbfgs' is efficient and preferred for L2 and multinomial.
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
        # Calculate variance to measure sensitivity (higher variance = higher sensitivity)
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
    plt.title("Sensitivity to C (Multiclass) - HAR")
    plt.legend()
    plt.grid(True)
    plt.savefig("har_lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close() # Use close to prevent display issues in some environments
    
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
        # Calculate variance (simple difference for two points)
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
    plt.title(f'L1 vs L2 Penalty Comparison (C={C_fixed}) - HAR')
    plt.xticks(x + width / 2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("har_lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close() # Close plot
    
    return results, metric_variances, best_by_metric


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess using existing train/test split
    X_train_raw, X_test_raw, Y_train, Y_test = load_and_preprocess(
        "../smartphone_train.csv", 
        "../smartphone_test.csv", 
        "Activity"
    )
    
    # scale features
    X_train, X_test = scale_features(X_train_raw, X_test_raw)
    
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
    with open("lr_sensitivity_results_har.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("LOGISTIC REGRESSION HYPERPARAMETER SENSITIVITY ANALYSIS - HAR DATASET (MULTICLASS)\n")
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
        f.write(f"Plot saved as: har_lr_c_sensitivity_plot.png\n\n")
        
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
        f.write(f"Plot saved as: har_lr_penalty_comparison_plot.png\n\n")
        
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

    print("\nResults saved to: lr_sensitivity_results_har.txt")
    print("Plots saved as: har_lr_c_sensitivity_plot.png, har_lr_penalty_comparison_plot.png")