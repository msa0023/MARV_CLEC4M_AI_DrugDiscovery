import os
import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

IN_TRAIN  = "data/train_data.csv"
IN_TEST   = "data/test_data.csv"
OUT_TRAIN = "data/train_pca.csv"
OUT_TEST  = "data/test_pca.csv"
os.makedirs("results", exist_ok=True)

DESCRIPTOR_COLS = [
    "MolWt", "MolLogP", "MaxPartialCharge", "MinPartialCharge",
    "MaxEStateIndex", "MinEStateIndex", "FpDensityMorgan1", "qed",
    "NumValenceElectrons", "Chi0", "Chi3n", "BalabanJ",
    "NumHeteroatoms", "NumRotatableBonds", "RingCount", "TPSA",
    "NumAtoms", "NumHeavyAtoms", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticCarbocycles", "NumAromaticCarbocycles",
    "FractionCSP3", "NumHAcceptors", "NumHDonors",
]

train = pd.read_csv(IN_TRAIN)
test  = pd.read_csv(IN_TEST)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(train[DESCRIPTOR_COLS].values)
X_test_sc  = scaler.transform(test[DESCRIPTOR_COLS].values)

pca_full = PCA().fit(X_train_sc)
ev = np.cumsum(pca_full.explained_variance_ratio_)
pd.DataFrame({
    "PC": range(1, len(ev)+1),
    "Variance_Explained": pca_full.explained_variance_ratio_,
    "Cumulative": ev
}).to_csv("results/explained_variance.csv", index=False)
print(f"PC1: {pca_full.explained_variance_ratio_[0]*100:.1f}%  "
      f"PC2: {pca_full.explained_variance_ratio_[1]*100:.1f}%  "
      f"Cumulative (2 PCs): {ev[1]*100:.1f}%")

pca         = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca  = pca.transform(X_test_sc)

pd.DataFrame(X_train_pca, columns=["PC1","PC2"]).assign(
    Label=train["Label"].values).to_csv(OUT_TRAIN, index=False)
pd.DataFrame(X_test_pca, columns=["PC1","PC2"]).assign(
    Label=test["Label"].values).to_csv(OUT_TEST, index=False)

joblib.dump(scaler, "results/scaler.pkl")
joblib.dump(pca,    "results/pca.pkl")
print("PCA complete. Scaler and PCA model saved to results/")
