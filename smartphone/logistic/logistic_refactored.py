import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

# Add analysis directory to path
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

warnings.filterwarnings("ignore")

def load_and_preprocess(train_path, test_path, target_col):
    """Load train and test data"""
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    print(f"\nTrain class distribution:\n{y_train.value_counts()}")
    print(f"\nTest class distribution:\n{y_test.value_counts()}")
    
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    """Scale the features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
    print(f"Test set: {X_test_scaled.shape[0]} samples")
    
    return X_train_scaled, X_test_scaled

def calculate_all_metrics(y_true, y_pred):
    """Calculate comprehensive metrics for multiclass"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def test_C_sensitivity(X_train, X_test, y_train, y_test, logger, C_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    """Test sensitivity to C hyperparameter (regularization strength)"""
    results_list = []
    
    for C in C_values:
        clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='C',
            param_value=C,
            metrics=metrics
        )
        
        print(f"C={C}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, [r['accuracy'] for r in results_list], marker='o', label='Accuracy')
    plt.semilogx(C_values, [r['f1'] for r in results_list], marker='s', label='F1')
    plt.xlabel("C (Inverse Regularization Strength)")
    plt.ylabel("Score")
    plt.title("Logistic Regression Sensitivity to C - HAR")
    plt.legend()
    plt.grid(True)
    plt.savefig("har_lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_penalty_comparison(X_train, X_test, y_train, y_test, logger):
    """Test sensitivity to penalty type"""
    penalties = ['l1', 'l2']
    solvers = ['saga', 'lbfgs']  # l1 needs saga, l2 can use lbfgs
    results_list = []
    
    for penalty, solver in zip(penalties, solvers):
        clf = LogisticRegression(penalty=penalty, solver=solver, max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        metrics = calculate_all_metrics(y_test, y_pred)
        results_list.append(metrics)
        
        logger.log_experiment(
            param_name='penalty',
            param_value=penalty,
            metrics=metrics,
            additional_info={'solver': solver}
        )
        
        print(f"{penalty} ({solver}): Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, penalty in enumerate(penalties):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=penalty)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Logistic Regression Penalty Comparison - HAR')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("har_lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def main():
    # Initialize logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="har_smartphone",
        algorithm="logistic_regression"
    )
    
    # Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess("../smartphone_train.csv", "../smartphone_test.csv", "Activity")
    
    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    # Run tests
    print("\n" + "="*60)
    print("Running C sensitivity test...")
    print("="*60)
    C_results = test_C_sensitivity(X_train_scaled, X_test_scaled, y_train, y_test, logger)
    
    print("\n" + "="*60)
    print("Running penalty comparison...")
    print("="*60)
    penalty_results = test_penalty_comparison(X_train_scaled, X_test_scaled, y_train, y_test, logger)
    
    # Save all results
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: logistic_regression_sensitivity_results.json")
    print("  - Text file: logistic_regression_sensitivity_results.txt")
    print("  - Plots: har_lr_c_sensitivity_plot.png, har_lr_penalty_comparison_plot.png")
    print("="*80)

if __name__ == "__main__":
    main()
