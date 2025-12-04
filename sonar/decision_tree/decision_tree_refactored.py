import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import warnings

current_file = Path(__file__).resolve()
analysis_path = current_file.parent.parent.parent / 'analysis'
sys.path.insert(0, str(analysis_path))

from standardized_logger import StandardizedLogger

warnings.filterwarnings("ignore")

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    print(f"Original shape: {df.shape}")
    
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
    else:
        print("No missing values found")
    
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate rows, removing...")
        df = df.drop_duplicates()
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    y = y.map({'R': 0, 'M': 1})
    
    print(f"\nClass distribution:\n{y.value_counts()}")
    return X, y

def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test

def calculate_all_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }

def test_max_depth_sensitivity(X_train, X_test, y_train, y_test, logger, depth_values=list(range(1, 21))):
    results = []
    for depth in depth_values:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        metrics = calculate_all_metrics(y_test, clf.predict(X_test))
        results.append(metrics)
        logger.log_experiment('max_depth', depth, metrics)
        print(f"max_depth={depth}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(depth_values, [r['accuracy'] for r in results], marker='o', label='Accuracy')
    plt.plot(depth_values, [r['f1'] for r in results], marker='s', label='F1')
    plt.xlabel("Max Depth")
    plt.ylabel("Score")
    plt.title("Decision Tree Sensitivity to Max Depth - Sonar")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_depth_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def test_min_samples_split(X_train, X_test, y_train, y_test, logger, split_values=list(range(2, 21))):
    results = []
    for split in split_values:
        clf = DecisionTreeClassifier(min_samples_split=split, random_state=42)
        clf.fit(X_train, y_train)
        metrics = calculate_all_metrics(y_test, clf.predict(X_test))
        results.append(metrics)
        logger.log_experiment('min_samples_split', split, metrics)
        print(f"min_samples_split={split}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(split_values, [r['accuracy'] for r in results], marker='o', label='Accuracy')
    plt.plot(split_values, [r['f1'] for r in results], marker='s', label='F1')
    plt.xlabel("Min Samples Split")
    plt.ylabel("Score")
    plt.title("Decision Tree Sensitivity to Min Samples Split - Sonar")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_split_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def test_min_samples_leaf(X_train, X_test, y_train, y_test, logger, leaf_values=list(range(1, 21))):
    results = []
    for leaf in leaf_values:
        clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
        clf.fit(X_train, y_train)
        metrics = calculate_all_metrics(y_test, clf.predict(X_test))
        results.append(metrics)
        logger.log_experiment('min_samples_leaf', leaf, metrics)
        print(f"min_samples_leaf={leaf}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(leaf_values, [r['accuracy'] for r in results], marker='o', label='Accuracy')
    plt.plot(leaf_values, [r['f1'] for r in results], marker='s', label='F1')
    plt.xlabel("Min Samples Leaf")
    plt.ylabel("Score")
    plt.title("Decision Tree Sensitivity to Min Samples Leaf - Sonar")
    plt.legend()
    plt.grid(True)
    plt.savefig("dt_leaf_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def test_criterion(X_train, X_test, y_train, y_test, logger):
    criteria = ["gini", "entropy"]
    results = []
    for criterion in criteria:
        clf = DecisionTreeClassifier(criterion=criterion, random_state=42)
        clf.fit(X_train, y_train)
        metrics = calculate_all_metrics(y_test, clf.predict(X_test))
        results.append(metrics)
        logger.log_experiment('criterion', criterion, metrics)
        print(f"criterion={criterion}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    for i, criterion in enumerate(criteria):
        values = [results[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=criterion)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Decision Tree Criterion Comparison - Sonar')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("dt_criterion_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def main():
    logger = StandardizedLogger(output_dir=Path("."), dataset="sonar", algorithm="decision_tree")
    X, y = load_and_preprocess("../sonar.csv")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    print("\n" + "="*60 + "\nRunning max_depth sensitivity...\n" + "="*60)
    test_max_depth_sensitivity(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60 + "\nRunning min_samples_split sensitivity...\n" + "="*60)
    test_min_samples_split(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60 + "\nRunning min_samples_leaf sensitivity...\n" + "="*60)
    test_min_samples_leaf(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60 + "\nRunning criterion comparison...\n" + "="*60)
    test_criterion(X_train, X_test, y_train, y_test, logger)
    
    logger.save_all()
    print("\n" + "="*80)
    print("✓ All results saved!")
    print("="*80)

if __name__ == "__main__":
    main()
