import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, GraphDescriptors, EState, QED

INPUT  = "input.csv"
OUTPUT = "output.csv"

DESCRIPTOR_NAMES = [
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
        return [np.nan] * len(DESCRIPTOR_NAMES)
    try:
        AllChem.ComputeGasteigerCharges(mol)
        pc = [a.GetDoubleProp("_GasteigerCharge") for a in mol.GetAtoms()]
        return [
            Descriptors.MolWt(mol), Descriptors.MolLogP(mol),
            max(pc), min(pc),
            max(EState.EStateIndices(mol)), min(EState.EStateIndices(mol)),
            sum(AllChem.GetMorganFingerprintAsBitVect(mol, 1)) / mol.GetNumAtoms(),
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
        return [np.nan] * len(DESCRIPTOR_NAMES)

df   = pd.read_csv(INPUT)
desc = pd.DataFrame([compute_descriptors(s) for s in df["Smiles"]], columns=DESCRIPTOR_NAMES)
df   = pd.concat([df, desc], axis=1)
df.to_csv(OUTPUT, index=False)
print(f"Descriptors generated for {len(df)} compounds -> {OUTPUT}")
