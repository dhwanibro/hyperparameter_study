import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time

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
    """Calculate comprehensive metrics for BINARY classification"""
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='binary', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='binary', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='binary', zero_division=0)
    }
    if Y_proba is not None:
        try:
            # for binary classification, use the probability of the positive class (column 1)
            metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
        except Exception as e:
            metrics['auc'] = 0.0
            print(f"AUC calculation failed: {e}")
    else:
        metrics['auc'] = 0.0
    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, C_values=[0.001, 0.01, 0.1, 1, 10, 100]):
    """
    Test sensitivity to regularization strength C
    C = 1/lambda (inverse regularization strength)
    """
    results_list = []
    c_results = {}
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for C in C_values:
        # Logistic Regression is binary by default, solver='liblinear' is a good robust choice for small datasets
        model = LogisticRegression(C=C, max_iter=2000, solver='liblinear', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results_list.append(test_metrics)
        c_results[C] = {'train': train_metrics, 'test': test_metrics}
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance for each metric
    metric_variances = {}
    for metric in metrics_to_check:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    # find best performing C for different metrics
    best_by_metric = {}
    for metric in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = C_values[best_idx]
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(C_values, values, marker="o", label=metric.upper())
    
    plt.xscale("log") # Use log scale for C values
    plt.xlabel("Regularization Strength C (log scale)")
    plt.ylabel("Score")
    plt.title("Sensitivity to C (Multiple Metrics) - Diabetes")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.show() # Can comment this out if running in non-interactive environment
    
    return c_results, metric_variances, best_by_metric

def test_penalty_types(X_train, X_test, Y_train, Y_test, C_fixed=1.0):
    """
    Test L1 vs L2 penalty with fixed C
    L1 (lasso) → feature selection, sets weights to zero
    L2 (ridge) → weight shrinkage, keeps all features
    """
    penalties = ['l1', 'l2']
    results = {}
    results_list = []
    metrics_to_check = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for penalty in penalties:
        # 'liblinear' solver supports both L1 and L2 penalties for binary classification
        model = LogisticRegression(penalty=penalty, C=C_fixed, max_iter=2000, solver='liblinear', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        results[penalty] = {'train': train_metrics, 'test': test_metrics}
        results_list.append(test_metrics)
        
        print(f"{penalty.upper()}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # calculate variance (difference) for each metric
    metric_variances = {}
    for metric in metrics_to_check:
        values = [r[metric] for r in results_list]
        # Variance for two values is simple: np.var([a, b]) = ((a-b)^2)/4
        metric_variances[metric] = np.var(values)
    
    # find best performing penalty type for different metrics
    best_by_metric = {}
    for metric in metrics_to_check:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = penalties[best_idx]
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, penalty in enumerate(penalties):
        values = [results[penalty]['test'][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=penalty.upper())
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'L1 vs L2 Penalty Comparison (C={C_fixed}, All Metrics) - Diabetes')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.show() # Can comment this out if running in non-interactive environment
    
    return results, metric_variances, best_by_metric


if __name__ == "__main__":
    # Load and preprocess
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
    print("Running C (Regularization) sensitivity test...")
    print("="*60)
    c_results, c_variances, best_c = test_C_sensitivity(X_train, X_test, Y_train, Y_test)
    
    print("\n" + "="*60)
    print("Running L1 vs L2 penalty comparison...")
    print("="*60)
    penalty_results, penalty_variances, best_penalty = test_penalty_types(X_train, X_test, Y_train, Y_test)
    
    # write everything to single txt file
    with open("lr_sensitivity_results_diabetes.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("LOGISTIC REGRESSION HYPERPARAMETER SENSITIVITY ANALYSIS - DIABETES DATASET\n")
        f.write("="*80 + "\n\n")
        
        # TEST 1: C sensitivity
        f.write("TEST 1: C SENSITIVITY (Regularization Strength)\n")
        f.write("-"*80 + "\n\n")
        f.write("Individual C value results:\n")
        for C, metrics in c_results.items():
            test = metrics['test']
            f.write(f"C = {C}:\n")
            f.write(f"  Test Metrics: Acc={test['accuracy']:.4f}, Prec={test['precision']:.4f}, "
                   f"Rec={test['recall']:.4f}, F1={test['f1']:.4f}, AUC={test['auc']:.4f}\n")
        
        f.write(f"\nVariance for each metric (over different C values):\n")
        for metric, var in c_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest C by metric:\n")
        for metric, c_val in best_c.items():
            f.write(f"  {metric}: {c_val}\n")
        f.write(f"Plot saved as: lr_c_sensitivity_plot.png\n\n")
        
        f.write("WHY: C controls the inverse of the regularization strength. Small C means strong\n")
        f.write("regularization (simpler model, mitigates overfitting), large C means weak regularization\n")
        f.write("(model fits data closely, risks overfitting). This balances bias-variance tradeoff.\n\n")
        
        # TEST 2: Penalty type
        f.write("TEST 2: L1 vs L2 PENALTY COMPARISON (C=1.0)\n")
        f.write("-"*80 + "\n\n")
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
        f.write(f"Plot saved as: lr_penalty_comparison_plot.png\n\n")
        
        f.write("WHY: L2 (Ridge) shrinks weights but keeps all features. L1 (Lasso) can drive weights to\n")
        f.write("exactly zero, effectively performing feature selection. Choice depends on feature importance.\n\n")
        
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
        f.write(f"\n  For ACCURACY (overall correctness):\n")
        f.write(f"    C_value: {best_c['accuracy']}\n")
        f.write(f"    penalty_type: {best_penalty['accuracy'].upper()}\n")
        
        f.write(f"\n  For PRECISION (true positives / predicted positives):\n")
        f.write(f"    C_value: {best_c['precision']}\n")
        f.write(f"    penalty_type: {best_penalty['precision'].upper()}\n")
        
        f.write(f"\n  For RECALL (true positives / actual positives):\n")
        f.write(f"    C_value: {best_c['recall']}\n")
        f.write(f"    penalty_type: {best_penalty['recall'].upper()}\n")
        
        f.write(f"\n  For F1-SCORE (balance of precision and recall):\n")
        f.write(f"    C_value: {best_c['f1']}\n")
        f.write(f"    penalty_type: {best_penalty['f1'].upper()}\n")
        
        f.write(f"\n  For AUC (Area Under Curve):\n")
        f.write(f"    C_value: {best_c['auc']}\n")
        f.write(f"    penalty_type: {best_penalty['auc'].upper()}\n")
        
    print("\nResults saved to: lr_sensitivity_results_diabetes.txt")
    print("Plots saved as: lr_c_sensitivity_plot.png, lr_penalty_comparison_plot.png")