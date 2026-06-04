import pandas as pd
from rdkit import Chem

INPUT  = "input.csv"
OUTPUT = "output.csv"

df = pd.read_csv(INPUT)
print(f"Loaded: {len(df)} compounds")

def is_valid(smi):
    try:
        return Chem.MolFromSmiles(smi) is not None
    except:
        return False

df = df.dropna(subset=["Smiles", "Label"])
df = df[df["Smiles"].apply(is_valid)]
df = df.drop_duplicates(subset=["Smiles"])
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv(OUTPUT, index=False)
print(f"Cleaned: {len(df)} compounds -> {OUTPUT}")
print(f"  Active (1): {(df['Label']==1).sum()}  |  Inactive (0): {(df['Label']==0).sum()}")
