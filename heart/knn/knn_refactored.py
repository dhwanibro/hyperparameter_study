import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

# Add analysis directory to path
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

warnings.filterwarnings("ignore")

def load_and_preprocess(filepath, target_col):
    """Load data and preprocess"""
    df = pd.read_csv(filepath)
    
    print(f"Original shape: {df.shape}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
        print(f"Shape after dropping NaN: {df.shape}")
    else:
        print("No missing values found")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
        print(f"Shape after dropping duplicates: {df.shape}")
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Convert {1,2} → {0,1}
    y = y.replace({1: 0, 2: 1})
    
    print(f"\nClass distribution:\n{y.value_counts()}")
    print(f"Class proportions:\n{y.value_counts(normalize=True)}")
    
    return X, y

def split_and_scale(X, y, test_size=0.2, random_state=42):
    """Split and scale the data"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def calculate_all_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def test_k_sensitivity(X_train, X_test, y_train, y_test, logger, k_values=list(range(1, 26))):
    """Test sensitivity to k hyperparameter"""
    results_list = []
    
    for k in k_values:
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='k',
            param_value=k,
            metrics=metrics
        )
        
        print(f"k={k}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [r['accuracy'] for r in results_list], marker='o', label='Accuracy')
    plt.plot(k_values, [r['f1'] for r in results_list], marker='s', label='F1')
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Score")
    plt.title("KNN Sensitivity to K - Heart Disease")
    plt.legend()
    plt.grid(True)
    plt.savefig("knn_k_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_weights_comparison(X_train, X_test, y_train, y_test, logger, best_k=5):
    """Test sensitivity to weights parameter"""
    weights_list = ["uniform", "distance"]
    results_list = []
    
    for weight in weights_list:
        clf = KNeighborsClassifier(n_neighbors=best_k, weights=weight)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='weights',
            param_value=weight,
            metrics=metrics,
            additional_info={'k': best_k}
        )
        
        print(f"weights={weight}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, weight in enumerate(weights_list):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=weight)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('KNN Weights Comparison - Heart Disease')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_weights_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_metric_comparison(X_train, X_test, y_train, y_test, logger, best_k=5):
    """Test sensitivity to distance metric"""
    metrics_list_param = ["euclidean", "manhattan", "minkowski"]
    results_list = []
    
    for metric in metrics_list_param:
        clf = KNeighborsClassifier(n_neighbors=best_k, metric=metric)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        perf_metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(perf_metrics)
        
        logger.log_experiment(
            param_name='metric',
            param_value=metric,
            metrics=perf_metrics,
            additional_info={'k': best_k}
        )
        
        print(f"metric={metric}: Acc={perf_metrics['accuracy']:.4f}, F1={perf_metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, metric in enumerate(metrics_list_param):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=metric)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('KNN Distance Metric Comparison - Heart Disease')
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("knn_metric_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def main():
    # Initialize logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="heart",
        algorithm="knn"
    )
    
    # Load and preprocess
    X, y = load_and_preprocess("../heart.csv", "target")
    
    # Split and scale
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    # Run tests
    print("\n" + "="*60)
    print("Running k sensitivity test...")
    print("="*60)
    k_results = test_k_sensitivity(X_train, X_test, y_train, y_test, logger)
    
    # Find best k based on F1 score
    best_k = max(range(1, 26), key=lambda k: k_results[k-1]['f1'])
    print(f"\nBest k: {best_k} (F1={k_results[best_k-1]['f1']:.4f})")
    
    print("\n" + "="*60)
    print("Running weights comparison...")
    print("="*60)
    weights_results = test_weights_comparison(X_train, X_test, y_train, y_test, logger, best_k)
    
    print("\n" + "="*60)
    print("Running distance metric comparison...")
    print("="*60)
    metric_results = test_metric_comparison(X_train, X_test, y_train, y_test, logger, best_k)
    
    # Save all results
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: knn_sensitivity_results.json")
    print("  - Text file: knn_sensitivity_results.txt")
    print("  - Plots: knn_k_sensitivity_plot.png, knn_weights_comparison_plot.png, knn_metric_comparison_plot.png")
    print("="*80)

if __name__ == "__main__":
    main()
