import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
# Updated imports for comprehensive binary classification metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ### The target variable is "NObeyesdad" (obesity level)
# Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III
# This is a MULTI-CLASS classification problem. AUC (for multi-class) and binary metrics will be adjusted or omitted/not calculated for simplicity.

def load_and_preprocess(filepath, target_col):
    """Load data and do preprocessing"""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    # check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()  # drop rows with missing values
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
    # columns like Gender, SMOKE, CAEC, CALC, MTRANS are categorical
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)  # don't encode target yet
    
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
    # if OG data has imbalance then test and train data has same imbalance ratio using stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # creating an instance of the standardscaler class
    # scales features to mean=0, std=1 so extreme values don't distort training
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    # the scaler object stores the mean and std of every column inside itself. transform just applies it
    
    # applying the train data's mean and sd to this, will not fit it cause that causes leakage
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics for MULTI-CLASS classification (Obesity Data)"""
    # For multi-class, we use 'macro' or 'weighted' averaging. 'Weighted' accounts for imbalance.
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    }
    # ROC AUC is more complex for multi-class and often requires one-vs-rest (ovr) or one-vs-one (ovo)
    # For simplicity and consistency, we will calculate AUC using 'ovr' method.
    if Y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(Y_true, Y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            metrics['auc'] = 0.0
            print(f"AUC calculation failed: {e}")
    else:
        metrics['auc'] = 0.0

    return metrics

def test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test, k_values=[1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 50]):
    """
    Test sensitivity to n_neighbors (k) hyperparameter
    """
    results_list = []
    k_results = {}
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        k_results[k] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"k={k}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in metrics_to_check:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing k for different metrics
    best_by_metric = {}
    for metric in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = k_values[best_idx]
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(k_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("n_neighbors (k)")
    plt.ylabel("Score")
    plt.title("Sensitivity to n_neighbors (Multiple Metrics) - Obesity")
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_k_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    # plt.show()
    
    return k_results, metric_variances, best_by_metric

def test_weights_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    """
    Test sensitivity to weights hyperparameter
    """
    weights_options = ['uniform', 'distance']
    results = {}
    results_list = []
    
    for weight in weights_options:
        model = KNeighborsClassifier(n_neighbors=k, weights=weight)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results[weight] = {'train': train_metrics, 'test': test_metrics}
        results_list.append(test_metrics)
        
        print(f"{weight}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for metric in metrics_to_check:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing weight for different metrics
    best_by_metric = {}
    for metric in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = weights_options[best_idx]
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, weight in enumerate(weights_options):
        values = [results[weight]['test'][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=weight)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Weights Comparison (k={k}, All Metrics) - Obesity')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_weights_comparison_plot.png", dpi=150, bbox_inches='tight')
    # plt.show()
    
    return results, metric_variances, best_by_metric

def test_metric_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    """
    Test sensitivity to metric (distance metric) hyperparameter
    """
    metrics_list = ['euclidean', 'manhattan']
    results = {}
    results_list = []
    
    for metric in metrics_list:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results[metric] = {'train': train_metrics, 'test': test_metrics}
        results_list.append(test_metrics)
        
        print(f"{metric}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    for m in metrics_to_check:
        values = [r[m] for r in results_list]
        metric_variances[m] = np.var(values)
    
    # find best performing distance metric for different metrics
    best_by_metric = {}
    for m in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][m])
        best_by_metric[m] = metrics_list[best_idx]
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, dist_metric in enumerate(metrics_list):
        values = [results[dist_metric]['test'][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=dist_metric)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Distance Metric Comparison (k={k}, All Metrics) - Obesity')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_metric_comparison_plot.png", dpi=150, bbox_inches='tight')
    # plt.show()
    
    return results, metric_variances, best_by_metric

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Load and preprocess (assuming 'obesity.csv' is in the parent directory)
    X, Y = load_and_preprocess("../obesity.csv", "NObeyesdad")
    
    # Split and scale (CRITICAL for KNN!)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*60)
    print("Running n_neighbors (k) sensitivity test...")
    print("="*60)
    k_results, k_variances, best_k = test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running weights comparison...")
    print("="*60)
    weights_results, weights_variances, best_weights = test_weights_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running distance metric comparison...")
    print("="*60)
    metric_results, metric_variances, best_metric = test_metric_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("knn_sensitivity_results_obesity.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("KNN HYPERPARAMETER SENSITIVITY ANALYSIS - OBESITY DATASET (Multi-Class)\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: n_neighbors
        f.write("TEST 1: n_neighbors (k) SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual k value results:\n")
        for k, metrics in k_results.items():
            test = metrics['test']
            f.write(f"k = {k}:\n")
            f.write(f"  Test Metrics (Weighted Avg): Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC (ovr)={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in k_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest k by metric:\n")
        for metric, k in best_k.items():
            f.write(f"  {metric}: {k}\n")
        f.write(f"Plot saved as: knn_k_sensitivity_plot.png\n\n")
        
        f.write("WHY: k controls the number of neighbors used for voting. Low k (1-3) is sensitive to noise\n")
        f.write("and overfits, high k (30+) overly smooths decision boundaries and underfits. The optimal k\n")
        f.write("balances local sensitivity with generalization. This is the MOST important KNN parameter.\n\n")
        
        # TEST 2: weights
        f.write("TEST 2: WEIGHTS COMPARISON (k=5)\n")
        f.write("-"*80 + "\n\n")
        for weight, metrics in weights_results.items():
            test = metrics['test']
            f.write(f"{weight}:\n")
            f.write(f"  Test Metrics (Weighted Avg): Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC (ovr)={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in weights_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest weights by metric:\n")
        for metric, weight in best_weights.items():
            f.write(f"  {metric}: {weight}\n")
        f.write(f"Plot saved as: knn_weights_comparison_plot.png\n\n")
        
        f.write("WHY: 'uniform' treats all k neighbors equally, while 'distance' gives more weight to closer\n")
        f.write("neighbors. Distance weighting can help when data density varies across the feature space.\n\n")
        
        # TEST 3: distance metric
        f.write("TEST 3: DISTANCE METRIC COMPARISON (k=5)\n")
        f.write("-"*80 + "\n\n")
        for dist_metric, metrics in metric_results.items():
            test = metrics['test']
            f.write(f"{dist_metric}:\n")
            f.write(f"  Test Metrics (Weighted Avg): Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC (ovr)={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in metric_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest distance metric by metric:\n")
        for metric, dist in best_metric.items():
            f.write(f"  {metric}: {dist}\n")
        f.write(f"Plot saved as: knn_metric_comparison_plot.png\n\n")
        
        f.write("WHY: Euclidean (L2) measures straight-line distance, Manhattan (L1) measures city-block\n")
        f.write("distance. Manhattan is less sensitive to outliers. Choice depends on feature space geometry.\n\n")
        
        # SUMMARY
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # calculate average variance across all metrics for each parameter
        avg_variances = {
            'n_neighbors': np.mean(list(k_variances.values())),
            'weights': np.mean(list(weights_variances.values())),
            'metric': np.mean(list(metric_variances.values()))
        }
        
        most_sensitive = max(avg_variances, key=avg_variances.get)
        least_sensitive = min(avg_variances, key=avg_variances.get)
        
        f.write(f"Most sensitive parameter: {most_sensitive} (avg variance = {avg_variances[most_sensitive]:.6f})\n")
        f.write(f"Least sensitive parameter: {least_sensitive} (avg variance = {avg_variances[least_sensitive]:.6f})\n\n")
        
        f.write("Sensitivity ranking (by average variance across all metrics):\n")
        for param, var in sorted(avg_variances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {param}: {var:.6f}\n")
        
        f.write(f"\nBest hyperparameters by metric:\n")
        
        metrics_list_for_summary = ['accuracy', 'precision', 'recall', 'f1', 'auc']
        
        f.write(f"\n  For ACCURACY (overall correctness):\n")
        f.write(f"    n_neighbors: {best_k['accuracy']}\n")
        f.write(f"    weights: {best_weights['accuracy']}\n")
        f.write(f"    metric: {best_metric['accuracy']}\n")
        
        f.write(f"\n  For PRECISION (minimizing false positives, weighted):\n")
        f.write(f"    n_neighbors: {best_k['precision']}\n")
        f.write(f"    weights: {best_weights['precision']}\n")
        f.write(f"    metric: {best_metric['precision']}\n")
        
        f.write(f"\n  For RECALL (minimizing false negatives, weighted):\n")
        f.write(f"    n_neighbors: {best_k['recall']}\n")
        f.write(f"    weights: {best_weights['recall']}\n")
        f.write(f"    metric: {best_metric['recall']}\n")
        
        f.write(f"\n  For F1-SCORE (balance of precision and recall, weighted):\n")
        f.write(f"    n_neighbors: {best_k['f1']}\n")
        f.write(f"    weights: {best_weights['f1']}\n")
        f.write(f"    metric: {best_metric['f1']}\n")
        
        f.write(f"\n  For AUC (Area Under Curve, OVR/Weighted):\n")
        f.write(f"    n_neighbors: {best_k['auc']}\n")
        f.write(f"    weights: {best_weights['auc']}\n")
        f.write(f"    metric: {best_metric['auc']}\n")
        
    print("\nResults saved to: knn_sensitivity_results_obesity.txt")
    print("Plots saved as: knn_k_sensitivity_plot.png, knn_weights_comparison_plot.png, knn_metric_comparison_plot.png")
