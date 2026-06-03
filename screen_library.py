import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, GraphDescriptors, EState, QED
from sklearn.impute import SimpleImputer

INPUT  = "data/new_dataset.csv"
OUTPUT = "results/active_compounds.csv"

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

df     = pd.read_csv(INPUT)
smiles = df["Smiles"].dropna().tolist()
print(f"Screening {len(smiles)} compounds...")

desc    = np.array([compute_descriptors(s) for s in smiles])
imputer = SimpleImputer(strategy="mean")
desc    = imputer.fit_transform(desc)

scaler = joblib.load("results/scaler.pkl")
pca    = joblib.load("results/pca.pkl")
model  = joblib.load("results/models/RF_model.pkl")

X_pca       = pca.transform(scaler.transform(desc))
predictions = model.predict(X_pca)

active = pd.DataFrame({"Smiles": smiles})[predictions == 1]
active.to_csv(OUTPUT, index=False)
print(f"Screening complete: {len(active)} predicted active compounds -> {OUTPUT}")
