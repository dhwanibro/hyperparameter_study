import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
#               PATH SETUP
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # heart/logistic
DATA_PATH = os.path.join(BASE_DIR, "..", "heart.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "hyperparam_results")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
#               LOAD & CLEAN DATA
# ============================================================
df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# Convert {1,2} → {0,1}
y = y.replace({1: 0, 2: 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
#           HYPERPARAMETER SETTINGS
# ============================================================
penalties = ["l1", "l2", "elasticnet"]
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
l1_ratio = 0.5   # For elasticnet only

results = []

# Select solver based on penalty
def get_solver(penalty):
    if penalty == "l1":
        return "liblinear"
    elif penalty == "l2":
        return "lbfgs"
    elif penalty == "elasticnet":
        return "saga"
    else:
        return "lbfgs"

# ============================================================
#               GRID SEARCH LOOP
# ============================================================
for penalty in penalties:
    for C in C_values:
        solver = get_solver(penalty)

        try:
            if penalty == "elasticnet":
                model = LogisticRegression(
                    penalty=penalty,
                    solver=solver,
                    C=C,
                    l1_ratio=l1_ratio,
                    max_iter=5000,
                )
            else:
                model = LogisticRegression(
                    penalty=penalty,
                    solver=solver,
                    C=C,
                    max_iter=5000,
                )

            model.fit(X_train_scaled, y_train)
            preds = model.predict(X_test_scaled)

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            var = np.var(preds)

            results.append([penalty, C, acc, prec, rec, f1, var])

        except Exception as e:
            results.append([penalty, C, f"ERR: {e}", "-", "-", "-", "-"])

# Convert to DataFrame
results_df = pd.DataFrame(
    results,
    columns=["Penalty", "C", "Accuracy", "Precision", "Recall", "F1", "Variance"]
)

# Save results file
results_path = os.path.join(OUTPUT_DIR, "lr_hyperparam_results.txt")
results_df.to_string(open(results_path, "w"))
print(f"Saved hyperparameter results → {results_path}")

# ============================================================
#             PLOTS: SENSITIVITY TO C
# ============================================================
plt.figure(figsize=(8,5))
for penalty in penalties:
    subset = results_df[results_df["Penalty"] == penalty]
    subset_numeric = subset[subset["Accuracy"] != "-"]  # Avoid errors
    plt.plot(subset_numeric["C"], subset_numeric["Accuracy"], marker='o', label=penalty)

plt.xscale("log")
plt.xlabel("C value (log scale)")
plt.ylabel("Accuracy")
plt.title("Logistic Regression Sensitivity to C")
plt.legend()
plt.grid(True)

plot1_path = os.path.join(OUTPUT_DIR, "lr_c_sensitivity_plot.png")
plt.savefig(plot1_path)
plt.close()
print(f"Saved C sensitivity plot → {plot1_path}")

# ============================================================
#             PLOTS: PENALTY COMPARISON
# ============================================================
avg_scores = results_df.groupby("Penalty")["Accuracy"].mean()

plt.figure(figsize=(7,5))
plt.bar(avg_scores.index, avg_scores.values)
plt.xlabel("Penalty")
plt.ylabel("Average Accuracy")
plt.title("Penalty Comparison (Avg Accuracy)")
plt.grid(True, axis='y')

plot2_path = os.path.join(OUTPUT_DIR, "lr_penalty_comparison_plot.png")
plt.savefig(plot2_path)
plt.close()
print(f"Saved penalty comparison plot → {plot2_path}")

# ============================================================
#             PLOTS: VARIANCE
# ============================================================
plt.figure(figsize=(8,5))
for penalty in penalties:
    subset = results_df[results_df["Penalty"] == penalty]
    subset_numeric = subset[subset["Variance"] != "-"]
    plt.plot(subset_numeric["C"], subset_numeric["Variance"], marker='o', label=penalty)

plt.xscale("log")
plt.xlabel("C value")
plt.ylabel("Variance of Predictions")
plt.title("Prediction Variance vs C")
plt.legend()
plt.grid(True)

plot3_path = os.path.join(OUTPUT_DIR, "lr_variance_plot.png")
plt.savefig(plot3_path)
plt.close()
print(f"Saved variance plot → {plot3_path}")
