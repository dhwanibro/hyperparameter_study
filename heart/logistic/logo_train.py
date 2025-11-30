import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, precision_recall_curve
)
import seaborn as sns
import joblib
import os

# =====================
# 1. LOAD DATA
# =====================
df = pd.read_csv("../heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Fix labels: convert {1, 2} → {0, 1}
y = y.replace({1: 0, 2: 1})

# =====================
# 2. TRAIN-TEST SPLIT
# =====================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================
# 3. SCALING
# =====================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================
# 4. TRAIN MODEL
# =====================
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train_scaled, y_train)

# Save model + scaler
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# =====================
# 5. PREDICTIONS
# =====================
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# =====================
# 6. SAVE METRICS
# =====================
results = f"""
ACCURACY: {accuracy_score(y_test, y_pred)}
PRECISION: {precision_score(y_test, y_pred)}
RECALL: {recall_score(y_test, y_pred)}
F1 SCORE: {f1_score(y_test, y_pred)}
AUC SCORE: {auc(*roc_curve(y_test, y_proba)[:2]):.4f}
"""

with open("results.txt", "w") as f:
    f.write(results)

# =====================
# 7. ENSURE RESULTS FOLDER
# =====================
os.makedirs("results", exist_ok=True)

# =====================
# 8. CONFUSION MATRIX PLOT
# =====================
plt.figure(figsize=(6, 4))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# =====================
# 9. ROC CURVE PLOT
# =====================
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}", linewidth=2)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
plt.close()

# =====================
# 10. PRECISION–RECALL CURVE PLOT
# =====================
precision, recall, _ = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, linewidth=2)
plt.title("Precision–Recall Curve - Logistic Regression")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.tight_layout()
plt.savefig("results/pr_curve.png")
plt.close()

print("All results + 3 plots saved successfully!")
