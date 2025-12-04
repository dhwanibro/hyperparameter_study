import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time

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
        'accuracy': float(accuracy_score(Y_true, Y_pred)),
        'precision': float(precision_score(Y_true, Y_pred, average='weighted', zero_division=0)),
        'recall': float(recall_score(Y_true, Y_pred, average='weighted', zero_division=0)),
        'f1': float(f1_score(Y_true, Y_pred, average='weighted', zero_division=0))
    }
    if Y_proba is not None:
        try:
            metrics['auc'] = float(roc_auc_score(Y_true, Y_proba, multi_class='ovr', average='weighted'))
        except:
            metrics['auc'] = 0.0
    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger, C_values=[0.01, 0.1, 1, 10, 100]):
    """Test sensitivity to C hyperparameter"""
    results_list = []
    
    for C in C_values:
        print(f"\nTraining SVM with C={C}...")
        start_time = time.time()
        
        model = SVC(C=C, kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, Y_train)
        
        training_time = time.time() - start_time
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        n_support = int(model.n_support_.sum())
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='C',
            param_value=float(C),
            metrics=test_metrics,
            additional_info={
                'n_support_vectors': n_support,
                'training_time_seconds': float(training_time)
            }
        )
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
              f"AUC={test_metrics['auc']:.4f}, Support Vectors={n_support}, Time={training_time:.2f}s")
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        ax1.plot(range(len(C_values)), values, marker="o", label=metric.upper())
    
    ax1.set_xticks(range(len(C_values)))
    ax1.set_xticklabels([str(c) for c in C_values])
    ax1.set_xlabel("C (Regularization Parameter)")
    ax1.set_ylabel("Score")
    ax1.set_title("Sensitivity to C (Performance Metrics) - HAR")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig("svm_C_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_kernel_sensitivity(X_train, X_test, Y_train, Y_test, logger):
    """Test sensitivity to kernel type"""
    kernels = ['linear', 'rbf', 'poly']
    results_list = []
    
    for kern in kernels:
        print(f"\nTraining SVM with kernel={kern}...")
        start_time = time.time()
        
        model = SVC(kernel=kern, C=1.0, probability=True, random_state=42)
        
        try:
            model.fit(X_train, Y_train)
            training_time = time.time() - start_time
            
            Y_test_pred = model.predict(X_test)
            Y_test_proba = model.predict_proba(X_test)
            
            test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
            results_list.append(test_metrics)
            
            n_support = int(model.n_support_.sum())
            
            # Log to standardized logger
            logger.log_experiment(
                param_name='kernel',
                param_value=kern,
                metrics=test_metrics,
                additional_info={
                    'n_support_vectors': n_support,
                    'training_time_seconds': float(training_time)
                }
            )
            
            print(f"{kern}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
                  f"AUC={test_metrics['auc']:.4f}, Support Vectors={n_support}, Time={training_time:.2f}s")
        
        except Exception as e:
            print(f"Error with kernel {kern}: {e}")
            dummy_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            results_list.append(dummy_metrics)
            
            # Still log the failure
            logger.log_experiment(
                param_name='kernel',
                param_value=kern,
                metrics=dummy_metrics,
                additional_info={'error': str(e)}
            )
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, kern in enumerate(kernels):
        if i < len(results_list):
            values = [results_list[i][m] for m in metrics_to_plot]
            plt.bar(x + i*width, values, width, label=kern)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Kernel Comparison (All Metrics) - HAR')
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("svm_kernel_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="smartphone_har",
        algorithm="svm"
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
    print("Running C (regularization) sensitivity test...")
    print("="*60)
    C_results = test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running kernel comparison...")
    print("="*60)
    kernel_results = test_kernel_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: svm_sensitivity_results.json")
    print("  - Text file: svm_sensitivity_results.txt")
    print("  - Plots: svm_C_sensitivity_plot.png, svm_kernel_comparison_plot.png")
    print("="*80)
