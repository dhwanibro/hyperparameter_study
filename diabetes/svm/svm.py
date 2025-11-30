import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time


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
    unique, counts = np.unique(Y, return_counts=True)
    for val, count in zip(unique, counts):
        print(f"Class {val}: {count} ({count/len(Y)*100:.2f}%)")
    
    return X, Y

def split_and_scale(X, Y):
    """Split data and scale features"""
    # if OG data has imbalance then test and train data has same imbalance ratio using stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # creating an instance of the standardscaler class
    # scales features to mean=0, std=1 so extreme values don't distort training
    # CRITICAL for KNN since it uses distance metrics!
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    # the scaler object stores the mean and std of every column inside itself. transform just applies it
    
    # applying the train data's mean and sd to this, will not fit it cause that causes leakage
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

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
            # for binary classification, use the probability of the positive class
            metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
        except:
            metrics['auc'] = 0.0
    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, C_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    """
    Test sensitivity to C hyperparameter (regularization parameter)
    C = inverse of regularization strength
    - Low C (0.001-0.1) → strong regularization → soft margin → prevents overfitting
      Allows more training errors, focuses on maximizing margin
    - High C (10-1000) → weak regularization → hard margin → can overfit
      Tries to classify all training points correctly, margin can be smaller
    
    C is usually THE MOST IMPORTANT hyperparameter for SVM
    Controls the bias-variance tradeoff directly
    """
    results_list = []
    C_results = {}
    training_times = {}
    
    for C in C_values:
        print(f"\nTraining SVM with C={C}...")
        start_time = time.time()
        
        # Using RBF kernel (default) which is good for non-linear problems
        # probability=True enables predict_proba() for AUC calculation
        # random_state=42 for reproducibility
        model = SVC(C=C, kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, Y_train)
        
        training_time = time.time() - start_time
        training_times[C] = training_time
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        # Store number of support vectors (indicator of model complexity)
        n_support = model.n_support_.sum()
        
        results_list.append(test_metrics)
        C_results[C] = {
            'train': train_metrics, 
            'test': test_metrics,
            'n_support': n_support,
            'training_time': training_time
        }
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
              f"AUC={test_metrics['auc']:.4f}, Support Vectors={n_support}, Time={training_time:.2f}s")
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing C for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = C_values[best_idx]
    
    # plot multiple metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance metrics
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        ax1.plot(range(len(C_values)), values, marker="o", label=metric.upper())
    
    ax1.set_xticks(range(len(C_values)))
    ax1.set_xticklabels([str(c) for c in C_values])
    ax1.set_xlabel("C (Regularization Parameter)")
    ax1.set_ylabel("Score")
    ax1.set_title("Sensitivity to C (Performance Metrics)")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')
    
    # Plot 2: Number of support vectors (model complexity indicator)
    support_vectors = [C_results[c]['n_support'] for c in C_values]
    ax2.plot(range(len(C_values)), support_vectors, marker="s", color='red', linewidth=2)
    ax2.set_xticks(range(len(C_values)))
    ax2.set_xticklabels([str(c) for c in C_values])
    ax2.set_xlabel("C (Regularization Parameter)")
    ax2.set_ylabel("Number of Support Vectors")
    ax2.set_title("Model Complexity vs C")
    ax2.grid(True)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig("svm_C_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return C_results, metric_variances, best_by_metric, training_times

def test_kernel_sensitivity(X_train, X_test, Y_train, Y_test):
    """
    Test sensitivity to kernel type
    kernel = the function that transforms data into higher dimensions
    
    - 'linear': No transformation, fastest, good for linearly separable data
      Decision boundary: hyperplane in original space
      
    - 'rbf' (Radial Basis Function): Most popular for non-linear problems
      Creates circular/elliptical decision boundaries
      Similarity measure: exp(-gamma * ||x1 - x2||^2)
      
    - 'poly' (Polynomial): Creates polynomial decision boundaries
      Can model interactions between features
      Complexity controlled by 'degree' parameter
      
    - 'sigmoid': Similar to neural network activation
      S-shaped decision boundaries
      Less commonly used than RBF
    
    Kernel choice is CRITICAL - wrong kernel = poor performance regardless of other params
    """
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results = {}
    results_list = []
    training_times = {}
    
    for kern in kernels:
        print(f"\nTraining SVM with kernel={kern}...")
        start_time = time.time()
        
        # Using default C=1.0, which is reasonable middle ground
        # poly kernel uses degree=3 by default
        model = SVC(kernel=kern, C=1.0, probability=True, random_state=42)
        
        try:
            model.fit(X_train, Y_train)
            training_time = time.time() - start_time
            training_times[kern] = training_time
            
            Y_train_pred = model.predict(X_train)
            Y_test_pred = model.predict(X_test)
            Y_test_proba = model.predict_proba(X_test)
            
            train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
            test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
            
            n_support = model.n_support_.sum()
            
            results[kern] = {
                'train': train_metrics, 
                'test': test_metrics,
                'n_support': n_support,
                'training_time': training_time
            }
            results_list.append(test_metrics)
            
            print(f"{kern}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
                  f"AUC={test_metrics['auc']:.4f}, Support Vectors={n_support}, Time={training_time:.2f}s")
        
        except Exception as e:
            print(f"Error with kernel {kern}: {e}")
            # Add dummy results if kernel fails
            dummy_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            results[kern] = {
                'train': dummy_metrics,
                'test': dummy_metrics,
                'n_support': 0,
                'training_time': 0
            }
            results_list.append(dummy_metrics)
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing kernel for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = kernels[best_idx]
    
    # grouped bar plot for multiple metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, kern in enumerate(kernels):
        values = [results[kern]['test'][m] for m in metrics_to_plot]
        ax1.bar(x + i*width, values, width, label=kern)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Kernel Comparison (All Metrics)')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Training time comparison
    times = [training_times.get(k, 0) for k in kernels]
    ax2.bar(kernels, times, color=['blue', 'green', 'orange', 'red'])
    ax2.set_xlabel('Kernel Type')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time by Kernel')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("svm_kernel_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, metric_variances, best_by_metric, training_times

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess using existing train/test split
    X, Y = load_and_preprocess("../diabetes_binary.csv", "Diabetes_binary")
    
    # scale features - CRITICAL for SVM performance
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*60)
    print("Running C (regularization) sensitivity test...")
    print("="*60)
    C_results, C_variances, best_C, C_times = test_C_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running kernel comparison...")
    print("="*60)
    kernel_results, kernel_variances, best_kernel, kernel_times = test_kernel_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("svm_sensitivity_results.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("SVM HYPERPARAMETER SENSITIVITY ANALYSIS - HAR DATASET\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: C parameter
        f.write("TEST 1: C (REGULARIZATION PARAMETER) SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual C results:\n")
        for C, metrics in C_results.items():
            test = metrics['test']
            train = metrics['train']
            f.write(f"C = {C}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
            f.write(f"  Train Metrics: Acc={train['accuracy']:.4f}, F1={train['f1']:.4f}\n")
            f.write(f"  Support Vectors: {metrics['n_support']}\n")
            f.write(f"  Training Time: {metrics['training_time']:.2f}s\n")
            f.write(f"  Overfitting Gap (Train-Test Acc): {train['accuracy']-test['accuracy']:.4f}\n\n")
        
        f.write(f"Variance for each metric:\n")
        for metric, var in C_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest C by metric:\n")
        for metric, C in best_C.items():
            f.write(f"  {metric}: {C}\n")
        f.write(f"\nPlot saved as: svm_C_sensitivity_plot.png\n\n")
        
        f.write("WHY: C controls the regularization strength (inverse relationship).\n")
        f.write("- Small C → Strong regularization → Soft margin → More training errors allowed\n")
        f.write("  → Simpler model → Better generalization → More support vectors\n")
        f.write("- Large C → Weak regularization → Hard margin → Fewer training errors\n")
        f.write("  → Complex model → Risk of overfitting → Fewer support vectors\n\n")
        f.write("This is typically the MOST IMPORTANT parameter for SVM performance.\n")
        f.write("Notice how support vectors decrease as C increases (model gets more strict).\n\n")
        
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
            f.write(f"  Support Vectors: {metrics['n_support']}\n")
            f.write(f"  Training Time: {metrics['training_time']:.2f}s\n")
            f.write(f"  Overfitting Gap (Train-Test Acc): {train['accuracy']-test['accuracy']:.4f}\n\n")
        
        f.write(f"Variance for each metric:\n")
        for metric, var in kernel_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest kernel by metric:\n")
        for metric, kern in best_kernel.items():
            f.write(f"  {metric}: {kern}\n")
        f.write(f"\nPlot saved as: svm_kernel_comparison_plot.png\n\n")
        
        f.write("WHY: Kernel choice determines how data is transformed into higher dimensions.\n")
        f.write("- linear: No transformation, creates linear decision boundary. Fastest, good for\n")
        f.write("  linearly separable data. Can underfit non-linear patterns.\n")
        f.write("- rbf: Radial Basis Function, most versatile. Creates smooth, non-linear boundaries.\n")
        f.write("  Usually best default choice for unknown data patterns.\n")
        f.write("- poly: Polynomial kernel, creates polynomial decision boundaries. Good for\n")
        f.write("  modeling feature interactions. Can be slow and numerically unstable.\n")
        f.write("- sigmoid: Similar to neural network activation. Less commonly used, can be\n")
        f.write("  unstable. Rarely the best choice.\n\n")
        f.write("Kernel choice is CRITICAL - wrong kernel = poor performance regardless of tuning.\n\n")
        
        # SUMMARY
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # calculate average variance across all metrics for each parameter
        avg_variances = {
            'C': np.mean(list(C_variances.values())),
            'kernel': np.mean(list(kernel_variances.values()))
        }
        
        most_sensitive = max(avg_variances, key=avg_variances.get)
        least_sensitive = min(avg_variances, key=avg_variances.get)
        
        f.write(f"Most sensitive parameter: {most_sensitive} (avg variance = {avg_variances[most_sensitive]:.6f})\n")
        f.write(f"Least sensitive parameter: {least_sensitive} (avg variance = {avg_variances[least_sensitive]:.6f})\n\n")
        
        f.write("Sensitivity ranking (by average variance across all metrics):\n")
        for param, var in sorted(avg_variances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {param}: {var:.6f}\n")
        
        f.write(f"\nBest hyperparameters by metric:\n")
        f.write(f"\n  For ACCURACY:\n")
        f.write(f"    C: {best_C['accuracy']}\n")
        f.write(f"    kernel: {best_kernel['accuracy']}\n")
        
        f.write(f"\n  For PRECISION:\n")
        f.write(f"    C: {best_C['precision']}\n")
        f.write(f"    kernel: {best_kernel['precision']}\n")
        
        f.write(f"\n  For RECALL:\n")
        f.write(f"    C: {best_C['recall']}\n")
        f.write(f"    kernel: {best_kernel['recall']}\n")
        
        f.write(f"\n  For F1:\n")
        f.write(f"    C: {best_C['f1']}\n")
        f.write(f"    kernel: {best_kernel['f1']}\n")
        
        f.write(f"\n  For AUC:\n")
        f.write(f"    C: {best_C['auc']}\n")
        f.write(f"    kernel: {best_kernel['auc']}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY INSIGHTS FOR SVM:\n")
        f.write("="*80 + "\n\n")
        f.write("1. FEATURE SCALING: SVM is extremely sensitive to feature scaling. Always use\n")
        f.write("   StandardScaler or similar before training.\n\n")
        f.write("2. SUPPORT VECTORS: The number of support vectors indicates model complexity.\n")
        f.write("   More support vectors = more complex decision boundary.\n\n")
        f.write("3. OVERFITTING CHECK: Compare train vs test accuracy. Large gap = overfitting.\n")
        f.write("   If overfitting: decrease C (stronger regularization) or use simpler kernel.\n\n")
        f.write("4. COMPUTATIONAL COST: RBF and poly kernels are slower than linear. Training time\n")
        f.write("   scales poorly with dataset size (roughly O(n^2) to O(n^3)).\n\n")
        f.write("5. HYPERPARAMETER TUNING ORDER:\n")
        f.write("   a) First: Choose kernel (try RBF as default)\n")
        f.write("   b) Second: Tune C (most impactful for performance)\n")
        f.write("   c) Third: If using RBF/poly, tune gamma (not done in this analysis)\n\n")
        
    print("\n" + "="*60)
    print("Results saved to: svm_sensitivity_results.txt")
    print("Plots saved as: svm_C_sensitivity_plot.png, svm_kernel_comparison_plot.png")
    print("="*60)