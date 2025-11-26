import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------------
# Paths & output names
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # heart/decision_tree
DATA_PATH = os.path.join(BASE_DIR, "..", "heart.csv")

OUT_DEPTH_PLOT = os.path.join(BASE_DIR, "dt_depth_sensitivity_plot.png")
OUT_LEAF_PLOT = os.path.join(BASE_DIR, "dt_leaf_sensitivity_plot.png")
OUT_SPLIT_PLOT = os.path.join(BASE_DIR, "dt_split_sensitivity_plot.png")
OUT_CRIT_PLOT = os.path.join(BASE_DIR, "dt_criterion_comparison_plot.png")
OUT_RESULTS_TXT = os.path.join(BASE_DIR, "dt_sensitivity_results.txt")

os.makedirs(BASE_DIR, exist_ok=True)

# ---------------------------
# Load & prepare data
# ---------------------------
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# convert {1,2} -> {0,1}
y = y.replace({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# helper to evaluate a model and return metrics dict
def eval_model(clf, X_test, y_test):
    preds = clf.predict(X_test)
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "Variance": float(np.var(preds))
    }

# ---------------------------
# Sensitivity: max_depth
# ---------------------------
depth_values = list(range(1, 21))  # 1..20
depth_results = []
for d in depth_values:
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)
    metrics = eval_model(clf, X_test, y_test)
    depth_results.append([d, metrics["Accuracy"], metrics["Precision"],
                          metrics["Recall"], metrics["F1"], metrics["Variance"]])

depth_df = pd.DataFrame(depth_results, columns=["max_depth", "Accuracy", "Precision", "Recall", "F1", "Variance"])

# Plot depth sensitivity (Accuracy vs depth)
plt.figure(figsize=(8,5))
plt.plot(depth_df["max_depth"], depth_df["Accuracy"], marker='o', label='Accuracy')
plt.plot(depth_df["max_depth"], depth_df["F1"], marker='s', label='F1')
plt.xlabel("max_depth")
plt.ylabel("Score")
plt.title("Decision Tree: Sensitivity to max_depth")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DEPTH_PLOT)
plt.close()

# ---------------------------
# Sensitivity: min_samples_leaf
# ---------------------------
leaf_values = [1,2,4,6,8,10]
leaf_results = []
for leaf in leaf_values:
    clf = DecisionTreeClassifier(min_samples_leaf=leaf, random_state=42)
    clf.fit(X_train, y_train)
    metrics = eval_model(clf, X_test, y_test)
    leaf_results.append([leaf, metrics["Accuracy"], metrics["Precision"],
                         metrics["Recall"], metrics["F1"], metrics["Variance"]])

leaf_df = pd.DataFrame(leaf_results, columns=["min_samples_leaf", "Accuracy", "Precision", "Recall", "F1", "Variance"])

# Plot leaf sensitivity (Accuracy vs leaf)
plt.figure(figsize=(8,5))
plt.plot(leaf_df["min_samples_leaf"], leaf_df["Accuracy"], marker='o', label='Accuracy')
plt.plot(leaf_df["min_samples_leaf"], leaf_df["F1"], marker='s', label='F1')
plt.xlabel("min_samples_leaf")
plt.ylabel("Score")
plt.title("Decision Tree: Sensitivity to min_samples_leaf")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_LEAF_PLOT)
plt.close()

# ---------------------------
# Sensitivity: min_samples_split
# ---------------------------
split_values = [2,5,10,20]
split_results = []
for split in split_values:
    clf = DecisionTreeClassifier(min_samples_split=split, random_state=42)
    clf.fit(X_train, y_train)
    metrics = eval_model(clf, X_test, y_test)
    split_results.append([split, metrics["Accuracy"], metrics["Precision"],
                          metrics["Recall"], metrics["F1"], metrics["Variance"]])

split_df = pd.DataFrame(split_results, columns=["min_samples_split", "Accuracy", "Precision", "Recall", "F1", "Variance"])

# Plot split sensitivity (Accuracy vs min_samples_split)
plt.figure(figsize=(8,5))
plt.plot(split_df["min_samples_split"], split_df["Accuracy"], marker='o', label='Accuracy')
plt.plot(split_df["min_samples_split"], split_df["F1"], marker='s', label='F1')
plt.xlabel("min_samples_split")
plt.ylabel("Score")
plt.title("Decision Tree: Sensitivity to min_samples_split")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_SPLIT_PLOT)
plt.close()

# ---------------------------
# Criterion comparison (gini, entropy, log_loss)
# ---------------------------
criteria = ["gini", "entropy", "log_loss"]
crit_results = []
for crit in criteria:
    # some sklearn versions may warn about log_loss with small data; keep using it per your sample
    clf = DecisionTreeClassifier(criterion=crit, random_state=42)
    clf.fit(X_train, y_train)
    metrics = eval_model(clf, X_test, y_test)
    crit_results.append([crit, metrics["Accuracy"], metrics["Precision"],
                         metrics["Recall"], metrics["F1"], metrics["Variance"]])

crit_df = pd.DataFrame(crit_results, columns=["criterion", "Accuracy", "Precision", "Recall", "F1", "Variance"])

# Plot criterion comparison (bar of Accuracy and F1)
x = np.arange(len(crit_df["criterion"]))
width = 0.35

plt.figure(figsize=(8,5))
plt.bar(x - width/2, crit_df["Accuracy"], width, label="Accuracy")
plt.bar(x + width/2, crit_df["F1"], width, label="F1")
plt.xticks(x, crit_df["criterion"])
plt.xlabel("Criterion")
plt.title("Decision Tree: Criterion Comparison (Accuracy vs F1)")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(OUT_CRIT_PLOT)
plt.close()

# ---------------------------
# Save combined results to text file
# ---------------------------
with open(OUT_RESULTS_TXT, "w") as f:
    f.write("Decision Tree sensitivity results\n")
    f.write("="*40 + "\n\n")

    f.write("Max depth sweep (1..20):\n")
    f.write(depth_df.to_string(index=False))
    f.write("\n\n" + "-"*40 + "\n\n")

    f.write("Min samples leaf sweep:\n")
    f.write(leaf_df.to_string(index=False))
    f.write("\n\n" + "-"*40 + "\n\n")

    f.write("Min samples split sweep:\n")
    f.write(split_df.to_string(index=False))
    f.write("\n\n" + "-"*40 + "\n\n")

    f.write("Criterion comparison:\n")
    f.write(crit_df.to_string(index=False))
    f.write("\n\n")

print("Decision tree sensitivity experiments complete.")
print(f"Results text saved to: {OUT_RESULTS_TXT}")
print("Plots saved:")
print(f" - {OUT_DEPTH_PLOT}")
print(f" - {OUT_LEAF_PLOT}")
print(f" - {OUT_SPLIT_PLOT}")
print(f" - {OUT_CRIT_PLOT}")
