import sys
import numpy as np
import pandas as pd
from pathlib import Path
import re
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Add analysis directory to path to import logger
current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

def load_and_preprocess(data_path, target_col='label'):
    """Load and preprocess SMS spam data"""
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values:\n{missing[missing > 0]}")
        df = df.dropna()
    else:
        print("No missing values")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
    
    if df[target_col].dtype == 'object':
        unique_labels = df[target_col].unique()
        print(f"Unique labels: {unique_labels}")
        if 'spam' in unique_labels or 'ham' in unique_labels:
            df[target_col] = df[target_col].map({'spam': 1, 'ham': 0})
        else:
            label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
            df[target_col] = df[target_col].map(label_map)
    
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

def test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test, logger, depth_values=[1, 2, 3, 5, 7, 10, 15, 20, None]):
    """Test sensitivity to max_depth hyperparameter"""
    results_list = []
    
    for depth in depth_values:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='max_depth',
            param_value=depth,
            metrics=test_metrics
        )
        
        print(f"Depth={depth}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    plt.figure(figsize=(12, 6))
    x_labels = [str(d) for d in depth_values]
    x_positions = range(len(depth_values))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(x_positions, values, marker="o", label=metric.upper())
    
    plt.xticks(x_positions, x_labels)
    plt.xlabel("max_depth")
    plt.ylabel("Score")
    plt.title("Sensitivity to max_depth (Multiple Metrics) - SMS Spam")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_depth_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test, logger, split_values=[2, 5, 10, 20, 50, 100]):
    """Test sensitivity to min_samples_split hyperparameter"""
    results_list = []
    
    for split in split_values:
        model = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='min_samples_split',
            param_value=split,
            metrics=test_metrics
        )
        
        print(f"split={split}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(split_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("min_samples_split")
    plt.ylabel("Score")
    plt.title("Sensitivity to min_samples_split (Multiple Metrics) - SMS Spam")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_split_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test, logger, leaf_values=[1, 2, 5, 10, 20, 50]):
    """Test sensitivity to min_samples_leaf hyperparameter"""
    results_list = []
    
    for leaf in leaf_values:
        model = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='min_samples_leaf',
            param_value=leaf,
            metrics=test_metrics
        )
        
        print(f"leaf={leaf}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    plt.figure(figsize=(12, 6))
    
    for metric in ['accuracy', 'f1', 'auc']:
        values = [r[metric] for r in results_list]
        plt.plot(leaf_values, values, marker="o", label=metric.upper())
    
    plt.xlabel("min_samples_leaf")
    plt.ylabel("Score")
    plt.title("Sensitivity to min_samples_leaf (Multiple Metrics) - SMS Spam")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_leaf_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

def test_criterion_sensitivity(X_train, X_test, Y_train, Y_test, logger):
    """Test sensitivity to criterion (splitting criterion)"""
    criteria = ['gini', 'entropy', 'log_loss']
    results_list = []
    
    for crit in criteria:
        model = DecisionTreeClassifier(criterion=crit, random_state=42)
        model.fit(X_train, Y_train)
        
        Y_test_pred = model.predict(X_test)
        Y_test_proba = model.predict_proba(X_test)
        
        test_metrics = calculate_all_metrics(Y_test, Y_test_pred, Y_test_proba)
        results_list.append(test_metrics)
        
        # Log to standardized logger
        logger.log_experiment(
            param_name='criterion',
            param_value=crit,
            metrics=test_metrics
        )
        
        print(f"{crit}: Acc={test_metrics['accuracy']:.4f}, F1={test_metrics['f1']:.4f}, AUC={test_metrics['auc']:.4f}")
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, crit in enumerate(criteria):
        values = [results_list[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=crit)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Criterion Comparison (All Metrics) - SMS Spam')
    plt.xticks(x + width, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("dt_criterion_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return results_list

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    # Initialize standardized logger
    logger = StandardizedLogger(
        output_dir=Path("."),
        dataset="sms_spam",
        algorithm="decision_tree"
    )
    
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
    print("Running max_depth sensitivity test...")
    print("="*60)
    depth_results = test_max_depth_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running min_samples_split sensitivity test...")
    print("="*60)
    split_results = test_min_samples_split_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running min_samples_leaf sensitivity test...")
    print("="*60)
    leaf_results = test_min_samples_leaf_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    print("\n" + "="*60)
    print("Running criterion comparison...")
    print("="*60)
    criterion_results = test_criterion_sensitivity(X_train, X_test, Y_train, Y_test, logger)
    
    # Save all results using standardized logger
    logger.save_all()
    
    print("\n" + "="*80)
    print("✓ All results saved using StandardizedLogger!")
    print("  - JSON file: decision_tree_sensitivity_results.json")
    print("  - Text file: decision_tree_sensitivity_results.txt")
    print("  - Plots: dt_depth_sensitivity_plot.png, dt_split_sensitivity_plot.png,")
    print("           dt_leaf_sensitivity_plot.png, dt_criterion_comparison_plot.png")
    print("="*80)
