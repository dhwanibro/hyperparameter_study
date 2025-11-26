import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from preprocess_sms import load_sms_dataset, preprocess_sms

# Function to evaluate sensitivity of a single hyperparameter
def evaluate_classifier(X, y, cls, hyperparam, values):
    print(f"\nRunning sensitivity for {cls.__name__} - {hyperparam}")
    results = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for v in values:
        print(f"Testing {hyperparam} = {v}")
        clf = cls(**{hyperparam: v})
        scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy', n_jobs=-1)
        results.append({'value': v, 'mean': scores.mean(), 'std': scores.std()})
        print(f"Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return pd.DataFrame(results)

# Function to generate and save a plot
def plot_sensitivity(df, hyperparam, out_path):
    plt.figure(figsize=(7,5))
    plt.errorbar(df['value'], df['mean'], yerr=df['std'], marker='o')
    plt.xlabel(hyperparam, fontsize=12)
    plt.ylabel("Mean CV Accuracy", fontsize=12)
    plt.title(f"Sensitivity for {hyperparam}", fontsize=14)
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")

# MAIN EXECUTION
if __name__ == "__main__":
    print("Loading SMS dataset...")
    df = load_sms_dataset("data/sms_spam.csv")
    X, y, vec = preprocess_sms(df)
    print("Dataset loaded and vectorized.")
    print("Shape of data:", X.shape)

    # Hyperparameter values to test
    C_values = [0.01, 0.1, 1, 5, 10]

    # Logistic Regression
    lr_results = evaluate_classifier(X, y, LogisticRegression, "C", C_values)
    lr_results.to_csv("results/lr_sensitivity.csv", index=False)
    print("Saved: results/lr_sensitivity.csv")
    plot_sensitivity(lr_results, "C", "plots/lr_sensitivity.png")

    # Linear SVM
    svm_results = evaluate_classifier(X, y, LinearSVC, "C", C_values)
    svm_results.to_csv("results/svm_sensitivity.csv", index=False)
    print("Saved:results/svm_sensitivity.csv")
    plot_sensitivity(svm_results, "C", "plots/svm_sensitivity.png")

    print("\nAll experiments completed successfully.")
