import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ============================================================
#                    PATH SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # heart/knn
DATA_PATH = os.path.join(BASE_DIR, "..", "heart.csv")

OUT_K_PLOT = os.path.join(BASE_DIR, "knn_k_sensitivity_plot.png")
OUT_METRIC_PLOT = os.path.join(BASE_DIR, "knn_metric_comparison_plot.png")
OUT_WEIGHTS_PLOT = os.path.join(BASE_DIR, "knn_weights_comparison_plot.png")
OUT_RESULTS_TXT = os.path.join(BASE_DIR, "knn_sensitivity_results.txt")

os.makedirs(BASE_DIR, exist_ok=True)


# ============================================================
#                    LOAD & CLEAN DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# Convert {1,2} → {0,1}
y = y.replace({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
#             Helper function to evaluate model
# ============================================================
def evaluate(clf):
    preds = clf.predict(X_test_scaled)
    return {
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, zero_division=0),
        "Recall": recall_score(y_test, preds, zero_division=0),
        "F1": f1_score(y_test, preds, zero_division=0),
        "Variance": float(np.var(preds))
    }


# ============================================================
#               1. K SENSITIVITY (k = 1 to 25)
# ============================================================
k_values = list(range(1, 25 + 1))
k_results = []

for k in k_values:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train_scaled, y_train)
    metrics = evaluate(clf)
    k_results.append([
        k,
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"],
        metrics["Variance"]
    ])

k_df = pd.DataFrame(k_results, columns=["k", "Accuracy", "Precision", "Recall", "F1", "Variance"])


# ---- Plot: K sensitivity ----
plt.figure(figsize=(9,5))
plt.plot(k_df["k"], k_df["Accuracy"], marker='o', label="Accuracy")
plt.plot(k_df["k"], k_df["F1"], marker='s', label="F1")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Score")
plt.title("KNN Sensitivity to K value")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_K_PLOT)
plt.close()


# ============================================================
#               2. WEIGHTS COMPARISON
# ============================================================
weights = ["uniform", "distance"]
weight_results = []

for w in weights:
    clf = KNeighborsClassifier(n_neighbors=5, weights=w)
    clf.fit(X_train_scaled, y_train)
    metrics = evaluate(clf)
    weight_results.append([
        w,
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"],
        metrics["Variance"]
    ])

weight_df = pd.DataFrame(weight_results, columns=["Weights", "Accuracy", "Precision", "Recall", "F1", "Variance"])


# ---- Plot: Weights comparison ----
plt.figure(figsize=(7,5))
plt.bar(weight_df["Weights"], weight_df["Accuracy"], label="Accuracy")
plt.bar(weight_df["Weights"], weight_df["F1"], alpha=0.6, label="F1")
plt.title("KNN Weight Comparison (Accuracy vs F1)")
plt.xlabel("Weights")
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_WEIGHTS_PLOT)
plt.close()


# ============================================================
#               3. METRIC COMPARISON
# ============================================================
metrics_list = ["euclidean", "manhattan", "minkowski"]
metric_results = []

for m in metrics_list:
    clf = KNeighborsClassifier(n_neighbors=5, metric=m)
    clf.fit(X_train_scaled, y_train)
    metrics = evaluate(clf)
    metric_results.append([
        m,
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"],
        metrics["Variance"]
    ])

metric_df = pd.DataFrame(metric_results, columns=["Metric", "Accuracy", "Precision", "Recall", "F1", "Variance"])


# ---- Plot: Metric comparison ----
plt.figure(figsize=(8,5))
plt.bar(metric_df["Metric"], metric_df["Accuracy"], label="Accuracy")
plt.bar(metric_df["Metric"], metric_df["F1"], alpha=0.6, label="F1")
plt.title("KNN Metric Comparison (Accuracy vs F1)")
plt.xlabel("Distance Metric")
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.savefig(OUT_METRIC_PLOT)
plt.close()


# ============================================================
#               4. SAVE ALL RESULTS TO TEXT FILE
# ============================================================
with open(OUT_RESULTS_TXT, "w") as f:
    f.write("KNN Sensitivity Results\n")
    f.write("="*45 + "\n\n")

    f.write("K sweep (1–25):\n")
    f.write(k_df.to_string(index=False))
    f.write("\n\n" + "-"*45 + "\n\n")

    f.write("Weights comparison (k=5):\n")
    f.write(weight_df.to_string(index=False))
    f.write("\n\n" + "-"*45 + "\n\n")

    f.write("Metric comparison (k=5):\n")
    f.write(metric_df.to_string(index=False))
    f.write("\n\n")

print("KNN sensitivity experiments complete!")
print(f"Saved results → {OUT_RESULTS_TXT}")
print("Saved plots:")
print(f" - {OUT_K_PLOT}")
print(f" - {OUT_WEIGHTS_PLOT}")
print(f" - {OUT_METRIC_PLOT}")
