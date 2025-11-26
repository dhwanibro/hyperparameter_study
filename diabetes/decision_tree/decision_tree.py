import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# ### The target variable is "Diabetes_binary"
# 0 = no diabetes
# 1 = diabetes

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
    # if OG data has imbalance then test and train data has same imbalance ratio using stratify
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # NOTE: Decision trees don't need scaling but we do it for consistency
    # Trees make splits based on feature values, not distances
    # Scaling doesn't hurt though
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
        metrics['auc'] = roc_auc_score(Y_true, Y_proba)
    return metrics

def test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test, depth_values=[1, 2, 3, 5, 7, 10, 15, 20, None]):
    """
    Test sensitivity to max_depth hyperparameter
    max_depth = maximum depth of the tree
    - Low depth (1-3) → shallow tree → underfitting → high bias
    - High depth (20+) → deep tree → overfitting → high variance
    - None → tree grows until all leaves are pure (usually overfits)
    
    Depth controls model complexity
    """
    results_list = []
    depth_results = {}
    
    for depth in depth_values:
        # Create decision tree classifier
        # random_state=42 makes results reproducible
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, Y_train)
        
        # predict() takes FEATURES (X), not labels (Y)
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)[:, 1]
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        depth_results[depth] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"Depth={depth}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing depth for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = depth_values[best_idx]
    
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
    plt.title("Sensitivity to max_depth (Multiple Metrics)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_depth_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return depth_results, metric_variances, best_by_metric

def test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test, split_values=[2, 5, 10, 20, 50, 100]):
    """
    Test sensitivity to min_samples_split hyperparameter
    min_samples_split = minimum number of samples required to split an internal node
    - Low values (2-5) → allows more splits → complex tree → can overfit
    - High values (50+) → fewer splits → simpler tree → can underfit
    
    This controls how "eager" the tree is to split nodes
    """
    results_list = []
    split_results = {}
    
    for split in split_values:
        # min_samples_split must be at least 2
        model = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)[:, 1]
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        split_results[split] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"split={split}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing split for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = split_values[best_idx]
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(split_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    plt.title("Sensitivity to min_samples_split (Multiple Metrics)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_split_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return split_results, metric_variances, best_by_metric

def test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test, leaf_values=[1, 2, 5, 10, 20, 50]):
    """
    Test sensitivity to min_samples_leaf hyperparameter
    min_samples_leaf = minimum number of samples required to be at a leaf node
    - Low values (1-2) → allows tiny leaves → can overfit to noise
    - High values (20+) → forces bigger leaves → smoother decision boundary
    
    This is another way to prevent overfitting
    """
    results_list = []
    leaf_results = {}
    
    for leaf in leaf_values:
        model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)[:, 1]
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        leaf_results[leaf] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"leaf={leaf}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing leaf for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = leaf_values[best_idx]
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(leaf_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Score")
    plt.title("Sensitivity to min_samples_leaf (Multiple Metrics)")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_leaf_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return leaf_results, metric_variances, best_by_metric

def test_criterion_sensitivity(X_train, X_test, Y_train, Y_test):
    """
    Test sensitivity to criterion (splitting criterion)
    criterion = the function to measure split quality
    - 'gini': Gini impurity (default) → measures how often a random sample would be misclassified
    - 'entropy': Information gain → measures information disorder
    - 'log_loss': Log loss, also known as cross-entropy
    
    Usually gini and entropy give similar results
    Gini is faster to compute, entropy might be slightly more accurate
    """
    criteria = ['gini', 'entropy', 'log_loss']
    results = {}
    results_list = []
    
    for crit in criteria:
        model = DecisionTreeClassifier(criterion=crit, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)[:, 1]
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results[crit] = {'train': train_metrics, 'test': test_metrics}
        results_list.append(test_metrics)
        
        print(f"{crit}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing criterion for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = criteria[best_idx]
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, crit in enumerate(criteria):
        values = [results[crit]['test'][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=crit)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Criterion Comparison (All Metrics)')
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("dt_criterion_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, metric_variances, best_by_metric

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # load and preprocess
    X, Y = load_and_preprocess("../diabetes_binary.csv", "Diabetes_binary")
    
    # split and scale
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*60)
    print("Running max_depth sensitivity test...")
    print("="*60)
    depth_results, depth_variances, best_depth = test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running min_samples_split sensitivity test...")
    print("="*60)
    split_results, split_variances, best_split = test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running min_samples_leaf sensitivity test...")
    print("="*60)
    leaf_results, leaf_variances, best_leaf = test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running criterion comparison...")
    print("="*60)
    criterion_results, criterion_variances, best_criterion = test_criterion_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("dt_sensitivity_results.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("DECISION TREE HYPERPARAMETER SENSITIVITY ANALYSIS - DIABETES DATASET\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: max_depth
        f.write("TEST 1: max_depth SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual max_depth results:\n")
        for depth, metrics in depth_results.items():
            test = metrics['test']
            f.write(f"max_depth = {depth}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in depth_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest max_depth by metric:\n")
        for metric, depth in best_depth.items():
            f.write(f"  {metric}: {depth}\n")
        f.write(f"Plot saved as: dt_depth_sensitivity_plot.png\n\n")
        
        f.write("WHY: max_depth controls tree complexity directly. Low values underfit (can't capture patterns),\n")
        f.write("high values overfit (memorize training noise). Sweet spot balances bias-variance tradeoff.\n")
        f.write("This is typically the MOST impactful parameter for decision trees.\n\n")
        
        # TEST 2: min_samples_split
        f.write("TEST 2: min_samples_split SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual min_samples_split results:\n")
        for split, metrics in split_results.items():
            test = metrics['test']
            f.write(f"min_samples_split = {split}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in split_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest min_samples_split by metric:\n")
        for metric, split in best_split.items():
            f.write(f"  {metric}: {split}\n")
        f.write(f"Plot saved as: dt_split_sensitivity_plot.png\n\n")
        
        f.write("WHY: Controls minimum samples to create a split. Lower values allow more granular splits,\n")
        f.write("higher values create smoother decision boundaries. Acts as regularization.\n\n")
        
        # TEST 3: min_samples_leaf
        f.write("TEST 3: min_samples_leaf SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual min_samples_leaf results:\n")
        for leaf, metrics in leaf_results.items():
            test = metrics['test']
            f.write(f"min_samples_leaf = {leaf}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in leaf_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest min_samples_leaf by metric:\n")
        for metric, leaf in best_leaf.items():
            f.write(f"  {metric}: {leaf}\n")
        f.write(f"Plot saved as: dt_leaf_sensitivity_plot.png\n\n")
        
        f.write("WHY: Enforces minimum leaf size, preventing overfitting to outliers. Similar effect to\n")
        f.write("min_samples_split but applied at leaves. Good for noisy datasets.\n\n")
        
        # TEST 4: criterion
        f.write("TEST 4: CRITERION COMPARISON\n")
        f.write("-"*80 + "\n\n")
        for crit, metrics in criterion_results.items():
            test = metrics['test']
            f.write(f"{crit}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric:\n")
        for metric, var in criterion_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest criterion by metric:\n")
        for metric, crit in best_criterion.items():
            f.write(f"  {metric}: {crit}\n")
        f.write(f"Plot saved as: dt_criterion_comparison_plot.png\n\n")
        
        f.write("WHY: Different impurity measures. Gini is computationally faster, entropy more theoretically\n")
        f.write("grounded. Usually minimal practical difference. Typically LEAST sensitive parameter.\n\n")
        
        # SUMMARY
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # calculate average variance across all metrics for each parameter
        avg_variances = {
            'max_depth': np.mean(list(depth_variances.values())),
            'min_samples_split': np.mean(list(split_variances.values())),
            'min_samples_leaf': np.mean(list(leaf_variances.values())),
            'criterion': np.mean(list(criterion_variances.values()))
        }
        
        most_sensitive = max(avg_variances, key=avg_variances.get)
        least_sensitive = min(avg_variances, key=avg_variances.get)
        
        f.write(f"Most sensitive parameter: {most_sensitive} (avg variance = {avg_variances[most_sensitive]:.6f})\n")
        f.write(f"Least sensitive parameter: {least_sensitive} (avg variance = {avg_variances[least_sensitive]:.6f})\n\n")
        
        f.write("Sensitivity ranking (by average variance across all metrics):\n")
        for param, var in sorted(avg_variances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {param}: {var:.6f}\n")
        
        f.write(f"\nBest hyperparameters by metric:\n")
        f.write(f"\n  For ACCURACY (overall correctness):\n")
        f.write(f"    max_depth: {best_depth['accuracy']}\n")
        f.write(f"    min_samples_split: {best_split['accuracy']}\n")
        f.write(f"    min_samples_leaf: {best_leaf['accuracy']}\n")
        f.write(f"    criterion: {best_criterion['accuracy']}\n")
        
        f.write(f"\n  For PRECISION (minimize false positives - when predicting diabetes, how often correct?):\n")
        f.write(f"    max_depth: {best_depth['precision']}\n")
        f.write(f"    min_samples_split: {best_split['precision']}\n")
        f.write(f"    min_samples_leaf: {best_leaf['precision']}\n")
        f.write(f"    criterion: {best_criterion['precision']}\n")
        
        f.write(f"\n  For RECALL (minimize false negatives - catch all actual diabetics):\n")
        f.write(f"    max_depth: {best_depth['recall']}\n")
        f.write(f"    min_samples_split: {best_split['recall']}\n")
        f.write(f"    min_samples_leaf: {best_leaf['recall']}\n")
        f.write(f"    criterion: {best_criterion['recall']}\n")
        
        f.write(f"\n  For F1 (balance precision & recall):\n")
        f.write(f"    max_depth: {best_depth['f1']}\n")
        f.write(f"    min_samples_split: {best_split['f1']}\n")
        f.write(f"    min_samples_leaf: {best_leaf['f1']}\n")
        f.write(f"    criterion: {best_criterion['f1']}\n")
        
        f.write(f"\n  For AUC (threshold-independent performance):\n")
        f.write(f"    max_depth: {best_depth['auc']}\n")
        f.write(f"    min_samples_split: {best_split['auc']}\n")
        f.write(f"    min_samples_leaf: {best_leaf['auc']}\n")
        f.write(f"    criterion: {best_criterion['auc']}\n")
        
    
    print("\n" + "="*60)
    print("Results saved to: dt_sensitivity_results.txt")
    print("Plots saved as: dt_depth_sensitivity_plot.png, dt_split_sensitivity_plot.png,")
    print("                dt_leaf_sensitivity_plot.png, dt_criterion_comparison_plot.png")
    print("="*60)