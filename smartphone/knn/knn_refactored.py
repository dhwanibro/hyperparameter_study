import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

def load_and_preprocess(train_path, test_path, target_col):
    """Load data and do preprocessing"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    missing_train = train_df.isnull().sum()
    if missing_train.sum() > 0:
        print(f"Missing values in train:\n{missing_train[missing_train > 0]}")
        train_df = train_df.dropna()
    else:
        print("No missing values in train")
    
    missing_test = test_df.isnull().sum()
    if missing_test.sum() > 0:
        print(f"Missing values in test:\n{missing_test[missing_test > 0]}")
        test_df = test_df.dropna()
    else:
        print("No missing values in test")
    
    duplicates_train = train_df.duplicated().sum()
    if duplicates_train > 0:
        print(f"Found {duplicates_train} duplicate rows in train, removing...")
        train_df = train_df.drop_duplicates()
    
    duplicates_test = test_df.duplicated().sum()
    if duplicates_test > 0:
        print(f"Found {duplicates_test} duplicate rows in test, removing...")
        test_df = test_df.drop_duplicates()
    
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
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

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

def test_n_neighbors_sensitivity(X_train, X_test, Y_train, Y_test, logger, k_values=[1, 3, 5, 7, 9, 11, 15, 20, 25, 30]):
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
    
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(k_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("n_neighbors (k)")
    plt.ylabel("Score")
    plt.title("Sensitivity to n_neighbors (Multiple Metrics) - HAR")
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
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, weight in enumerate(weights_options):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=weight)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Weights Comparison (k={k}, All Metrics) - HAR')
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
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, dist_metric in enumerate(metrics_list):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=dist_metric)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'Distance Metric Comparison (k={k}, All Metrics) - HAR')
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
        dataset="smartphone_har",
        algorithm="knn"
    )
    
    # Load and preprocess
    X_train, X_test, Y_train, Y_test = load_and_preprocess(
        "../smartphone_train.csv",
        "../smartphone_test.csv",
        "Activity"
    )
    
    # Scale features
    X_train, X_test = scale_features(X_train, X_test)
    
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
