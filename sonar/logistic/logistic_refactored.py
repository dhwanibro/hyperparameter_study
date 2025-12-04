import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
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

def test_C_sensitivity(X_train, X_test, y_train, y_test, logger, C_values=[0.001, 0.01, 0.1, 1, 10, 100, 1000]):
    results = []
    for C in C_values:
        clf = LogisticRegression(C=C, max_iter=10000, random_state=42)
        clf.fit(X_train, y_train)
        metrics = calculate_all_metrics(y_test, clf.predict(X_test))
        results.append(metrics)
        logger.log_experiment('C', C, metrics)
        print(f"C={C}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(C_values, [r['accuracy'] for r in results], marker='o', label='Accuracy')
    plt.semilogx(C_values, [r['f1'] for r in results], marker='s', label='F1')
    plt.xlabel("C (Inverse Regularization Strength)")
    plt.ylabel("Score")
    plt.title("Logistic Regression Sensitivity to C - Sonar")
    plt.legend()
    plt.grid(True)
    plt.savefig("lr_c_sensitivity_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def test_penalty(X_train, X_test, y_train, y_test, logger):
    penalties = ['l1', 'l2']
    solvers = ['liblinear', 'lbfgs']
    results = []
    for penalty, solver in zip(penalties, solvers):
        clf = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, random_state=42)
        clf.fit(X_train, y_train)
        metrics = calculate_all_metrics(y_test, clf.predict(X_test))
        results.append(metrics)
        logger.log_experiment('penalty', penalty, metrics, additional_info={'solver': solver})
        print(f"{penalty} ({solver}): Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")
    
    plt.figure(figsize=(10, 6))
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics_to_plot))
    width = 0.35
    for i, penalty in enumerate(penalties):
        values = [results[i][m] for m in metrics_to_plot]
        plt.bar(x + i*width, values, width, label=penalty)
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Logistic Regression Penalty Comparison - Sonar')
    plt.xticks(x + width/2, metrics_to_plot)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("lr_penalty_comparison_plot.png", dpi=150, bbox_inches='tight')
    plt.close()
    return results

def main():
    logger = StandardizedLogger(output_dir=Path("."), dataset="sonar", algorithm="logistic_regression")
    X, y = load_and_preprocess("../sonar.csv")
    X_train, X_test, y_train, y_test = split_and_scale(X, y)
    
    print("\n" + "="*60 + "\nRunning C sensitivity...\n" + "="*60)
    test_C_sensitivity(X_train, X_test, y_train, y_test, logger)
    
    print("\n" + "="*60 + "\nRunning penalty comparison...\n" + "="*60)
    test_penalty(X_train, X_test, y_train, y_test, logger)
    
    logger.save_all()
    print("\n" + "="*80)
    print("✓ All results saved!")
    print("="*80)

if __name__ == "__main__":
    main()
