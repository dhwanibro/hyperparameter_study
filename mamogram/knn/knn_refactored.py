import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

# ### Target variable: Severity (0 = benign, 1 = malignant)

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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
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
            metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
        except:
            metrics['auc'] = 0.0
    return metrics

def test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test, logger, k_values=[1, 3, 5, 7, 9, 11, 15, 20, 25, 30, 50]):
    """Test sensitivity to n_neighbors (k) hyperparameter"""
    results_list = []
    
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='n_neighbors',
            param_value=k,
            metrics=test_metrics
        )
        
        print(f"k={k}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(k_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("n_neighbors (k)")
    plt.ylabel("Score")
    plt.title("Sensitivity to n_neighbors (Multiple Metrics) - Mammogram")
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_k_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_weights_sensitivity(X_train, X_test, Y_train, Y_test, logger, k=5):
    """Test sensitivity to weights hyperparameter"""
    weights_options = ['uniform', 'distance']
    results_list = []
    
    for weight in weights_options:
        model = KNeighborsClassifier(n_neighbors=k, weights=weight)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='weights',
            param_value=weight,
            metrics=test_metrics,
            additional_info={'n_neighbors': k}
        )
        
        print(f"{weight}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, weight in enumerate(weights_options):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=weight)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Weights Comparison (k={k}, All Metrics) - Mammogram')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_weights_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_metric_sensitivity(X_train, X_test, Y_train, Y_test, logger, k=5):
    """Test sensitivity to metric (distance metric) hyperparameter"""
    metrics_list = ['euclidean', 'manhattan']
    results_list = []
    
    for metric in metrics_list:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='distance_metric',
            param_value=metric,
            metrics=test_metrics,
            additional_info={'n_neighbors': k}
        )
        
        print(f"{metric}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, dist_metric in enumerate(metrics_list):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=dist_metric)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Distance Metric Comparison (k={k}, All Metrics) - Mammogram')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_metric_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="mammogram",
        algorithm="knn"
    )
    
    # load and preprocess
    X, Y = load_and_preprocess("../mamogram.csv", "Severity")
    
    # split and scale (CRITICAL for KNN!)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
    print("\n" + "="*60)
    print("Running n_neighbors (k) sensitivity test...")
    print("="*60)
    k_results = test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running weights comparison...")
    print("="*60)
    weights_results = test_weights_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running distance metric comparison...")
    print("="*60)
    metric_results = test_metric_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: knn_sensitivity_results.json")
    print("  - Text file: knn_sensitivity_results.txt")
    print("  - Plots: knn_k_sensitivity_plot.png, knn_weights_comparison_plot.png,")
    print("           knn_metric_comparison_plot.png")
    print("="*80)
