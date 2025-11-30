import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
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

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, C_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    """Test sensitivity to C hyperparameter"""
    results_list = []
    C_results = {}
    training_times = {}
    
    for C in C_values:
        print(f"\nTraining SVM with C={C}...")
        start_time = time.time()
        
        model = SVC(C=C, kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, Y_train)
        
        training_time = time.time() - start_time
        training_times[C] = training_time
        
        Y_train_pred = model.predict(X_train)
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        train_metrics = calculate_all_metrics(Y_train, Y_train_pred)
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
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
    
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = C_values[best_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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
    """Test sensitivity to kernel type"""
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results = {}
    results_list = []
    training_times = {}
    
    for kern in kernels:
        print(f"\nTraining SVM with kernel={kern}...")
        start_time = time.time()
        
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
            dummy_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            results[kern] = {
                'train': dummy_metrics,
                'test': dummy_metrics,
                'n_support': 0,
                'training_time': 0
            }
            results_list.append(dummy_metrics)
    
    metric_variances = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        metric_variances[metric] = np.var(values)
    
    best_by_metric = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        best_idx = max(range(len(results_list)), key=lambda i: results_list[i][metric])
        best_by_metric[metric] = kernels[best_idx]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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
    
    # Test C parameter
    print("\n" + "="*60)
    print("Running C (regularization) sensitivity test...")
    print("="*60)
    C_results, C_variances, best_C, C_times = test_C_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # Test kernel
    print("\n" + "="*60)
    print("Running kernel comparison...")
    print("="*60)
    kernel_results, kernel_variances, best_kernel, kernel_times = test_kernel_sensitivity(X_train, X_test, Y_train, Y_test)
    
    # Save results
    with open("svm_sensitivity_results.txt", "w") as f:
        f.write("="*80 + "\n")
        f.write("SVM HYPERPARAMETER SENSITIVITY ANALYSIS - SMS SPAM DATASET\n")
        f.write("="*80 + "\n\n")
        
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
            f.write(f"  Overfitting Gap: {train['accuracy']-test['accuracy']:.4f}\n\n")
        
        f.write(f"Variance for each metric:\n")
        for metric, var in C_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest C by metric:\n")
        for metric, C in best_C.items():
            f.write(f"  {metric}: {C}\n")
        f.write(f"\nPlot saved as: svm_C_sensitivity_plot.png\n\n")
        
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
            f.write(f"  Overfitting Gap: {train['accuracy']-test['accuracy']:.4f}\n\n")
        
        f.write(f"Variance for each metric:\n")
        for metric, var in kernel_variances.items():
            f.write(f"  {metric}: {var:.6f}\n")
        f.write(f"\nBest kernel by metric:\n")
        for metric, kern in best_kernel.items():
            f.write(f"  {metric}: {kern}\n")
        f.write(f"\nPlot saved as: svm_kernel_comparison_plot.png\n\n")
        
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        avg_variances = {
            'C': np.mean(list(C_variances.values())),
            'kernel': np.mean(list(kernel_variances.values()))
        }
        
        most_sensitive = max(avg_variances, key=avg_variances.get)
        least_sensitive = min(avg_variances, key=avg_variances.get)
        
        f.write(f"Most sensitive parameter: {most_sensitive} (avg variance = {avg_variances[most_sensitive]:.6f})\n")
        f.write(f"Least sensitive parameter: {least_sensitive} (avg variance = {avg_variances[least_sensitive]:.6f})\n\n")
        
        f.write("Sensitivity ranking:\n")
        for param, var in sorted(avg_variances.items(), key=lambda x: x[1], reverse=True):
            f.write(f"  {param}: {var:.6f}\n")
        
        f.write(f"\nBest hyperparameters by metric:\n")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            f.write(f"\n  For {metric.upper()}:\n")
            f.write(f"    C: {best_C[metric]}\n")
            f.write(f"    kernel: {best_kernel[metric]}\n")
        
    print("\n" + "="*60)
    print("Results saved to: svm_sensitivity_results.txt")
    print("Plots saved as: svm_C_sensitivity_plot.png, svm_kernel_comparison_plot.png")
    print("="*60)