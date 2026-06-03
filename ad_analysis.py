import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, GraphDescriptors, EState, QED
from sklearn.impute import SimpleImputer

mpl.rcParams["svg.fonttype"] = "none"
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

def compute_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [np.nan] * len(DESCRIPTOR_COLS)
    try:
        AllChem.ComputeGasteigerCharges(mol)
        pc = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
        return [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            max(pc), min(pc),
            max(EState.EStateIndices(mol)), min(EState.EStateIndices(mol)),
            sum(AllChem.GetMorganFingerprintAsBitVect(mol,1))/mol.GetNumAtoms(),
            QED.qed(mol), Descriptors.NumValenceElectrons(mol),
            GraphDescriptors.Chi0(mol), GraphDescriptors.Chi3n(mol),
            GraphDescriptors.BalabanJ(mol),
            Descriptors.NumHeteroatoms(mol), Descriptors.NumRotatableBonds(mol),
            Descriptors.RingCount(mol), Descriptors.TPSA(mol),
            mol.GetNumAtoms(), Descriptors.HeavyAtomCount(mol),
            Descriptors.NumAromaticRings(mol), Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticCarbocycles(mol), Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.NumHAcceptors(mol), Descriptors.NumHDonors(mol),
        ]
    except:
        return [np.nan] * len(DESCRIPTOR_COLS)

train  = pd.read_csv("data/train_pca.csv")
X_tr   = train[["PC1","PC2"]].values
n, p   = X_tr.shape
h_star = 3 * (p + 1) / n

XtX_inv  = np.linalg.pinv(X_tr.T @ X_tr)
h_train  = np.array([x @ XtX_inv @ x for x in X_tr])
centroid = X_tr.mean(axis=0)
cov_inv  = np.linalg.pinv(np.cov(X_tr.T))
sd_train = np.array([np.sqrt((x-centroid) @ cov_inv @ (x-centroid)) for x in X_tr])

scaler = joblib.load("results/scaler.pkl")
pca    = joblib.load("results/pca.pkl")
df_new = pd.read_csv("data/new_dataset.csv")
smiles = df_new["Smiles"].dropna().tolist()
print(f"Computing descriptors for {len(smiles)} compounds...")

desc    = np.array([compute_descriptors(s) for s in smiles])
imputer = SimpleImputer(strategy="mean")
desc    = imputer.fit_transform(desc)
X_q     = pca.transform(scaler.transform(desc))

h_q  = np.array([x @ XtX_inv @ x for x in X_q])
sd_q = np.array([np.sqrt((x-centroid) @ cov_inv @ (x-centroid)) for x in X_q])

in_ad  = int(np.sum((h_q <= h_star) & (sd_q <= 3.0)))
out_ad = len(h_q) - in_ad

fig, ax = plt.subplots(figsize=(9, 6))
ax.scatter(h_train, sd_train, c="lightblue", edgecolors="steelblue",
           alpha=0.5, s=35, label=f"Training set (n={n})", zorder=2)
ax.scatter(h_q, sd_q, c="tomato", edgecolors="darkred",
           alpha=0.35, s=12, label=f"Phytochemical library (n={len(h_q)})", zorder=3)
ax.axvline(x=h_star, color="black", linestyle="--", linewidth=1.2, label=f"h* = {h_star:.3f}")
ax.axhline(y=3.0,    color="gray",  linestyle="--", linewidth=1.2, label="SD threshold = 3.0")
ax.set_xlabel("Leverage (h)", fontsize=12)
ax.set_ylabel("Standardized Mahalanobis Distance", fontsize=12)
ax.set_title("Applicability Domain — Williams Plot\n(Phytochemical Library vs Training Set)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=9)
ax.text(0.98, 0.98,
        f"Within AD : {in_ad}/{len(h_q)} ({100*in_ad/len(h_q):.1f}%)\n"
        f"Outside AD: {out_ad}/{len(h_q)} ({100*out_ad/len(h_q):.1f}%)",
        transform=ax.transAxes, fontsize=10, va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.tight_layout()
plt.savefig("results/AD_williams.svg")
plt.savefig("results/AD_williams.tiff", dpi=600)
plt.close()

pd.DataFrame([{
    "Total": len(h_q), "Within_AD": in_ad, "Outside_AD": out_ad,
    "Within_AD_%": f"{100*in_ad/len(h_q):.1f}%",
    "h_star": round(h_star, 4), "SD_threshold": 3.0
}]).to_csv("results/AD_summary.csv", index=False)

print(f"AD complete: {in_ad}/{len(h_q)} ({100*in_ad/len(h_q):.1f}%) within AD")
print("Saved: results/AD_williams.tiff, results/AD_summary.csv")
