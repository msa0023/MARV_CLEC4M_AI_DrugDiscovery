import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold, cross_val_predict

mpl.rcParams["svg.fonttype"] = "none"
os.makedirs("results", exist_ok=True)

MODEL_NAMES = ["kNN", "SVM", "RF", "NB", "GB"]
COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
X_train, y_train = train[["PC1","PC2"]].values, train["Label"].values
X_test,  y_test  = test[["PC1","PC2"]].values,  test["Label"].values

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (X, y, title, use_cv) in zip(axes, [
    (X_train, y_train, "Training Set (10-Fold CV)", True),
    (X_test,  y_test,  "Test Set",                  False),
]):
    for name, color in zip(MODEL_NAMES, COLORS):
        model = joblib.load(f"{name}_model.pkl")
        if use_cv:
            probs = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:,1]
        else:
            probs = model.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, probs)
        ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{name} (AUC = {auc(fpr,tpr):.2f})")

    ax.plot([0,1],[0,1], "k--", lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curves — {title}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("roc.svg")
plt.savefig("roc.tiff", dpi=600)
plt.close()
print("Saved: roc.tiff")
