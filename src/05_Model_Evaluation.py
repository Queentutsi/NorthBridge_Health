"""
05_Model_Evaluation.py
NorthBridge Health — SLA Breach Model Evaluation & NHS-Themed Visualisation
"""

import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

# NHS Colour Palette
NHS_BLUE = "#005EB8"
NHS_DARK_BLUE = "#003087"
NHS_LIGHT_BLUE = "#41B6E6"
NHS_YELLOW = "#FFB81C"
NHS_GREY = "#768692"
NHS_LIGHT_GREY = "#E8EDEE"

print("SCRIPT STARTED")

# -------------------------------------------------------------
# 1. Load metrics
# -------------------------------------------------------------

with open("models/model_metrics.json", "r") as f:
    metrics = json.load(f)

df_metrics = pd.DataFrame(metrics).T

print("\n📊 MODEL PERFORMANCE SUMMARY")
print(df_metrics)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# -------------------------------------------------------------
# 2. Bar Charts (Accuracy, Precision, Recall, F1, AUC)
# -------------------------------------------------------------

for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=df_metrics.index,
        y=df_metrics[metric],
        palette=[NHS_BLUE, NHS_DARK_BLUE, NHS_LIGHT_BLUE, NHS_YELLOW]
    )
    plt.title(f"{metric.upper()} Comparison", fontsize=14)
    plt.ylabel(metric)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"outputs/{metric}_comparison.png")
    plt.close()

# -------------------------------------------------------------
# 3. Confusion Matrices
# -------------------------------------------------------------

for model_name in metrics.keys():
    cm = np.array(metrics[model_name]["confusion_matrix"])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",   # FIXED — valid colormap
        cbar=False
    )
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_confusion_matrix.png")
    plt.close()

# -------------------------------------------------------------
# 4. ROC Curves — Correct Column Alignment
# -------------------------------------------------------------

DATA_FILE = r"C:\Users\akand\OneDrive\Documents\data journey\Amdari Resources\DS Projects\NorthBridge_Health\Data\processed\model_dataset.csv"
df = pd.read_csv(DATA_FILE)

X = df.drop(columns=["SLABreached"])
y = df["SLABreached"]

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# One-hot encode
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Load scaler
scaler = joblib.load("models/scaler.joblib")

# Load Logistic Regression to get expected columns
lr_model = joblib.load("models/LogisticRegression.joblib")

expected_cols = lr_model.coef_.shape[1]

# Align X to expected number of columns
X = X.reindex(columns=X.columns, fill_value=0)

# Scale numeric columns
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
X_scaled = X.copy()
X_scaled[numeric_cols] = scaler.transform(X_scaled[numeric_cols])

# -------------------------------------------------------------
# Generate ROC curves
# -------------------------------------------------------------

for model_name in metrics.keys():

    print(f"\nGenerating ROC curve for {model_name}...")

    model = joblib.load(f"models/{model_name}.joblib")

    if model_name == "LogisticRegression":
        probs = model.predict_proba(X_scaled)[:, 1]
    else:
        probs = model.predict_proba(X)[:, 1]

    fpr, tpr, _ = roc_curve(y, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color=NHS_BLUE, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color=NHS_GREY, lw=1, linestyle="--")
    plt.title(f"{model_name} — ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_roc_curve.png")
    plt.close()

# -------------------------------------------------------------
# 5. Feature Importance (RF + XGB)
# -------------------------------------------------------------

for model_name in ["RandomForest", "XGBoost"]:
    print(f"\nGenerating Feature Importance for {model_name}...")

    model = joblib.load(f"models/{model_name}.joblib")

    importances = model.feature_importances_
    feature_names = X.columns

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    sns.barplot(
        y=fi["feature"],
        x=fi["importance"],
        palette=[NHS_BLUE] * len(fi)
    )
    plt.title(f"{model_name} — Top 20 Feature Importances")
    plt.tight_layout()
    plt.savefig(f"outputs/{model_name}_feature_importance.png")
    plt.close()

print("\n🎨 All evaluation charts saved to /outputs/")
