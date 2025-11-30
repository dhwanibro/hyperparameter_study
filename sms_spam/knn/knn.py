import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


def load_and_preprocess(data_path, target_col='label'):
    """Load and preprocess SMS spam data"""
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values:\n{missing[missing > 0]}")
        df = df.dropna()
    else:
        print("No missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
    
    # Convert labels to binary if needed
    if df[target_col].dtype == 'object':
        unique_labels = df[target_col].unique()
        print(f"Unique labels: {unique_labels}")
        if 'spam' in unique_labels or 'ham' in unique_labels:
            df[target_col] = df[target_col].map({'spam': 1, 'ham': 0})
        else:
            # Assume first unique value is 0, second is 1
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            df[target_col] = df[target_col].map(label_map)
    
    # Check class distribution
    print(f"\nClass distribution:")
    print(df[target_col].value_counts())
    spam_count = df[target_col].sum()
    print(f"Class 1: {spam_count} ({spam_count/len(df)*100:.2f}%)")
    print(f"Class 0: {len(df)-spam_count} ({(len(df)-spam_count)/len(df)*100:.2f}%)")
    
    return df

def preprocess_text(text):
    """Clean and preprocess text"""
    # Lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def scale_features(X_train, X_test):
    """Convert text to TF-IDF features"""
    print(f"\nVectorizing text data...")
    
    # Preprocess text
    X_train_clean = X_train.apply(preprocess_text)
    X_test_clean = X_test.apply(preprocess_text)
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=3000, stop_words='english', min_df=2)
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)
    
    print(f"Train feature matrix shape: {X_train_vec.shape}")
    print(f"Test feature matrix shape: {X_test_vec.shape}")
    
    return X_train_vec, X_test_vec

def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics for binary classification"""
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, zero_division=0),
        'recall': recall_score(Y_true, Y_pred, zero_division=0),
        'f1': f1_score(Y_true, Y_pred, zero_division=0)
    }
    if Y_proba is not None:
        try:
            metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
        except:
            metrics['auc'] = 0.0
    return metrics

def split_and_scale(X, Y):
    """Split data and scale features (CRITICAL for Regularized Models like Logistic Regression)"""
    # use stratify to maintain class ratio
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # scales features to mean=0, std=1
    scaler = StandardScaler()
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test


def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics for multiclass"""
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

def test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test, k_values=[1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 50]):
    """
    Test sensitivity to n_neighbors (k) hyperparameter
    n_neighbors (k) = number of nearest neighbors to consider for voting
    - k=1 → uses only closest neighbor → very sensitive to noise → overfits
    - k=3-7 → often works well, good balance
    - k large (30+) → smooths decision boundary → may underfit
    
    Rule of thumb: k should be odd for binary classification to avoid ties
    k should be sqrt(n_samples) as starting point
    """
    results_list = []
    k_results = {}
    
    for k in k_values:
        # KNeighborsClassifier finds k nearest neighbors and does majority vote
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
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing k for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = k_values[best_idx]
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(k_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("n_neighbors (k)")
    plt.ylabel("Score")
    plt.title("Sensitivity to n_neighbors (Multiple Metrics)")
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_k_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return k_results, metric_variances, best_by_metric

def test_weights_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    """
    Test sensitivity to weights hyperparameter
    weights = how to weight the neighbors when voting
    - 'uniform': All neighbors vote equally (default)
    - 'distance': Closer neighbors have more influence
      → weight = 1 / distance
      → points that are closer matter more
    
    'distance' can help when you have varying density of data
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
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing weight for different metrics
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
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
    plt.title(f'Weights Comparison (k={k}, All Metrics)')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_weights_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, metric_variances, best_by_metric

def test_metric_sensitivity(X_train, X_test, Y_train, Y_test, k=5):
    """
    Test sensitivity to metric (distance metric) hyperparameter
    metric = how to calculate distance between points
    - 'euclidean': straight-line distance (default)
      → d = sqrt((x1-x2)^2 + (y1-y2)^2)
      → most common, works well generally
    - 'manhattan': city-block distance (sum of absolute differences)
      → d = |x1-x2| + |y1-y2|
      → less sensitive to outliers than euclidean
    
    Choice depends on data characteristics and feature space
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
    for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[m] for r in results_list]
        metric_variances[m] = np.var(values)
    
    # find best performing distance metric for different metrics
    best_by_metric = {}
    for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
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
    plt.title(f'Distance Metric Comparison (k={k}, All Metrics)')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_metric_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return results, metric_variances, best_by_metric

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    df = load_and_preprocess("../sms_spam.csv", target_col='label')
    
    # Assuming text column is named 'Message' - adjust if different
    text_col = 'Message' if 'Message' in df.columns else df.columns[1]
    target_col = 'Category' if 'Category' in df.columns else df.columns[0]
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        df[text_col], df[target_col], test_size=0.2, random_state=42, stratify=df[target_col]
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize text (replaces scale_features for text data)
    X_train, X_test = scale_features(X_train, X_test)
    
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
    with open("knn_sensitivity_results.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("KNN HYPERPARAMETER SENSITIVITY ANALYSIS - HAR DATASET\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: n_neighbors
        f.write("TEST 1: n_neighbors (k) SENSITIVITY\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual k value results:\n")
        for k, metrics in k_results.items():
            test = metrics['test']
            f.write(f"k = {k}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
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
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
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
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
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
        f.write(f"\n  For ACCURACY (overall correctness):\n")
        f.write(f"    n_neighbors: {best_k['accuracy']}\n")
        f.write(f"    weights: {best_weights['accuracy']}\n")
        f.write(f"    metric: {best_metric['accuracy']}\n")
        
        f.write(f"\n  For PRECISION (weighted average across all activity classes):\n")
        f.write(f"    n_neighbors: {best_k['precision']}\n")
        f.write(f"    weights: {best_weights['precision']}\n")
        f.write(f"    metric: {best_metric['precision']}\n")
        
        f.write(f"\n  For RECALL (weighted average - how well we catch each activity):\n")
        f.write(f"    n_neighbors: {best_k['recall']}\n")
        f.write(f"    weights: {best_weights['recall']}\n")
        f.write(f"    metric: {best_metric['recall']}\n")
        
        f.write(f"\n  For F1 (balance precision & recall):\n")
        f.write(f"    n_neighbors: {best_k['f1']}\n")
        f.write(f"    weights: {best_weights['f1']}\n")
        f.write(f"    metric: {best_metric['f1']}\n")
        
        f.write(f"\n  For AUC (threshold-independent performance, one-vs-rest):\n")
        f.write(f"    n_neighbors: {best_k['auc']}\n")
        f.write(f"    weights: {best_weights['auc']}\n")
        f.write(f"    metric: {best_metric['auc']}\n")
    
    print("\n" + "="*60)
    print("Results saved to: knn_sensitivity_results.txt")
    print("Plots saved as: knn_k_sensitivity_plot.png, knn_weights_comparison_plot.png,")
    print("                knn_metric_comparison_plot.png")
    print("="*60)