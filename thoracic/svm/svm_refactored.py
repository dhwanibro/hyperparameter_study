import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import time
import warnings

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

# Suppress warnings
warnings.filterwarnings("ignore")

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
    
    # encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)
    
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
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, Y_train, Y_test

def calculate_all_metrics(Y_true, Y_pred, Y_proba=None):
    """Calculate comprehensive metrics with fix for Binary AUC"""
    metrics = {
        'accuracy': accuracy_score(Y_true, Y_pred),
        'precision': precision_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'recall': recall_score(Y_true, Y_pred, average='weighted', zero_division=0),
        'f1': f1_score(Y_true, Y_pred, average='weighted', zero_division=0)
    }
    
    if Y_proba is not None:
        try:
            if Y_proba.shape[1] == 2:
                metrics['auc'] = roc_auc_score(Y_true, Y_proba[:, 1])
            else:
                metrics['auc'] = roc_auc_score(Y_true, Y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            metrics['auc'] = 0.0
            print(f"AUC calculation failed: {e}")
    else:
        metrics['auc'] = 0.0

    return metrics

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger, C_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    """Test sensitivity to C hyperparameter (regularization parameter)"""
    results_list = []
    training_times = []
    support_vectors = []
    
    for C in C_values:
        print(f"\nTraining SVM with C={C}...")
        start_time = time.time()
        
        model = SVC(C=C, kernel='rbf', probability=True, random_state=42)
        model.fit(X_train, Y_train)
        
        training_time = time.time() - start_time
        training_times.append(training_time)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        
        n_support = model.n_support_.sum()
        support_vectors.append(n_support)
        
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='C',
            param_value=C,
            metrics=test_metrics,
            additional_info={
                'n_support_vectors': int(n_support),
                'training_time': float(training_time)
            }
        )
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
              f"AUC={test_metrics['auc']:.4f}, Support Vectors={n_support}, Time={training_time:.2f}s")
    
    # plot multiple metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance metrics
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        ax1.plot(range(len(C_values)), values, marker="o", label=metric.upper())
    
    ax1.set_xticks(range(len(C_values)))
    ax1.set_xticklabels([str(c) for c in C_values])
    ax1.set_xlabel("C (Regularization Parameter)")
    ax1.set_ylabel("Score")
    ax1.set_title("Sensitivity to C (Performance Metrics) - Thoracic")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')
    
    # Plot 2: Number of support vectors
    ax2.plot(range(len(C_values)), support_vectors, marker="s", color='red', linewidth=2)
    ax2.set_xticks(range(len(C_values)))
    ax2.set_xticklabels([str(c) for c in C_values])
    ax2.set_xlabel("C (Regularization Parameter)")
    ax2.set_ylabel("Number of Support Vectors")
    ax2.set_title("Model Complexity vs C - Thoracic")
    ax2.grid(True)
    ax2.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig("svm_C_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_kernel_sensitivity(X_train, X_test, Y_train, Y_test, logger):
    """Test sensitivity to kernel type"""
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    results_list = []
    training_times = []
    support_vectors = []
    
    for kern in kernels:
        print(f"\nTraining SVM with kernel={kern}...")
        start_time = time.time()
        
        model = SVC(kernel=kern, C=1.0, probability=True, random_state=42)
        
        try:
            model.fit(X_train, Y_train)
            training_time = time.time() - start_time
            training_times.append(training_time)
            
            Y_test_pred = model.predict(X_test)
            Y_test_proba = model.predict_proba(X_test)
            
            test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
            
            n_support = model.n_support_.sum()
            support_vectors.append(n_support)
            
            results_list.append(test_metrics)
            
            # Log to standardized logger
            logger.log_experiment(
                param_name='kernel',
                param_value=kern,
                metrics=test_metrics,
                additional_info={
                    'n_support_vectors': int(n_support),
                    'training_time': float(training_time)
                }
            )
            
            print(f"{kern}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, "
                  f"AUC={test_metrics['auc']:.4f}, Support Vectors={n_support}, Time={training_time:.2f}s")
        
        except Exception as e:
            print(f"Error with kernel {kern}: {e}")
            dummy_metrics = {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auc': 0.0}
            results_list.append(dummy_metrics)
            training_times.append(0)
            support_vectors.append(0)
    
    # grouped bar plot for multiple metrics
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Performance comparison
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.2
    
    for i, kern in enumerate(kernels):
        values = [results_list[i][m] for m in metrics_to_plot]
        ax1.bar(x + i*width, values, width, label=kern)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Kernel Comparison (All Metrics) - Thoracic')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(metrics_to_plot)
    ax1.legend()
    ax1.grid(True, axis='y', alpha=0.3)
    
    # Plot 2: Training time comparison
    ax2.bar(kernels, training_times, color=['blue', 'green', 'orange', 'red'])
    ax2.set_xlabel('Kernel Type')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time by Kernel - Thoracic')
    ax2.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("svm_kernel_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="thoracic",
        algorithm="svm"
    )
    
    # Load and preprocess
    X, Y = load_and_preprocess("../ThoraricSurgery.csv", "Risk1Yr")
    
    # Split and scale (CRITICAL for SVM!)
    X_train, X_test, Y_train, Y_test = split_and_scale(X, Y)
    
    # run tests
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
