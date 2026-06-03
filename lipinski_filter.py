import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

INPUT  = "results/active_compounds.csv"
OUTPUT = "results/drug_like_compounds.csv"

def get_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MW"  : Descriptors.MolWt(mol),
        "HBD" : Lipinski.NumHDonors(mol),
        "HBA" : Lipinski.NumHAcceptors(mol),
        "LogP": Descriptors.MolLogP(mol),
    }

df    = pd.read_csv(INPUT)
props = pd.DataFrame(df["Smiles"].apply(get_properties).tolist())
df    = pd.concat([df, props], axis=1).dropna(subset=["MW"])

drug_like = df[
    (df["MW"]   < 500) &
    (df["HBD"] <= 5)   &
    (df["HBA"] <= 10)  &
    (df["LogP"] <= 5)
]

drug_like.to_csv(OUTPUT, index=False)
print(f"Input: {len(df)} | Passed Lipinski filter: {len(drug_like)} -> {OUTPUT}")
