import sys
import numpy as np
import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import warnings

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

# Suppress convergence warnings
warnings.filterwarnings("ignore")

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
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text

def scale_features(X_train, X_test):
    """Convert text to TF-IDF features"""
    print(f"\nVectorizing text data...")
    
    X_train_clean = X_train.apply(preprocess_text)
    X_test_clean = X_test.apply(preprocess_text)
    
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

def test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger, C_values=[0.001, 0.01, 0.1, 1, 10, 100]):
    """Test sensitivity to regularization strength C"""
    results_list = []
    
    for C in C_values:
        model = LogisticRegression(C=C, max_iter=2000, solver='liblinear', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='C',
            param_value=C,
            metrics=test_metrics
        )
        
        print(f"C={C}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # plot multiple metrics
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(C_values, values, marker="o", label=metric.upper())
    
    plt.xscale("log")
    plt.xlabel("Regularization Strength C (log scale)")
    plt.ylabel("Score")
    plt.title("Sensitivity to C (Multiple Metrics) - SMS Spam")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_penalty_types(X_train, X_test, Y_train, Y_test, logger, C_fixed=1.0):
    """Test L1 vs L2 penalty with fixed C"""
    penalties = ['l1', 'l2']
    results_list = []
    
    for penalty in penalties:
        model = LogisticRegression(penalty=penalty, C=C_fixed, max_iter=2000, solver='liblinear', random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='penalty',
            param_value=penalty,
            metrics=test_metrics,
            additional_info={'C': C_fixed}
        )
        
        print(f"{penalty.upper()}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    # grouped bar plot for multiple metrics
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    
    for i, penalty in enumerate(penalties):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=penalty.upper())
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title(f'L1 vs L2 Penalty Comparison (C={C_fixed}, All Metrics) - SMS Spam')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="sms_spam",
        algorithm="logistic_regression"
    )
    
    # Load and preprocess
    df = load_and_preprocess("../sms_spam.csv", target_col='label')
    
    text_col = 'Message' if 'Message' in df.columns else df.columns[1]
    target_col = 'Category' if 'Category' in df.columns else df.columns[0]
    
    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(
        df[text_col], df[target_col], test_size=0.2, random_state=42, stratify=df[target_col]
    )
    
    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Vectorize text
    X_train, X_test = scale_features(X_train, X_test)
    
    print("\n" + "="*60)
    print("Running C (Regularization) sensitivity test...")
    print("="*60)
    C_results = test_C_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running L1 vs L2 penalty comparison...")
    print("="*60)
    penalty_results = test_penalty_types(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: logistic_regression_sensitivity_results.json")
    print("  - Text file: logistic_regression_sensitivity_results.txt")
    print("  - Plots: lr_c_sensitivity_plot.png, lr_penalty_comparison_plot.png")
    print("="*80)
