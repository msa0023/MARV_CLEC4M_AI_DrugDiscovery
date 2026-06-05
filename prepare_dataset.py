import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

INPUT     = "input.csv"
OUT_TRAIN = "train.csv"
OUT_TEST  = "test.csv"

DESCRIPTOR_COLS = [
    "MolWt", "MolLogP", "MaxPartialCharge", "MinPartialCharge",
    "MaxEStateIndex", "MinEStateIndex", "FpDensityMorgan1", "qed",
    "NumValenceElectrons", "Chi0", "Chi3n", "BalabanJ",
    "NumHeteroatoms", "NumRotatableBonds", "RingCount", "TPSA",
    "NumAtoms", "NumHeavyAtoms", "NumAromaticRings", "NumSaturatedRings",
    "NumAliphaticCarbocycles", "NumAromaticCarbocycles",
    "FractionCSP3", "NumHAcceptors", "NumHDonors",
]

df  = pd.read_csv(INPUT)
X   = df[DESCRIPTOR_COLS]
y   = df["Label"]
idx = df.index

imputer = SimpleImputer(strategy="mean")
X_imp   = imputer.fit_transform(X)

X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_imp, y, idx, test_size=0.30, stratify=y, random_state=1
)

smote = SMOTE(k_neighbors=5, random_state=1)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

train_df = pd.DataFrame(X_train_bal, columns=DESCRIPTOR_COLS)
train_df.insert(0, "Label", y_train_bal)
train_df.to_csv(OUT_TRAIN, index=False)

test_df = pd.DataFrame(X_test, columns=DESCRIPTOR_COLS)
test_df.insert(0, "Smiles", df.loc[idx_test, "Smiles"].values)
test_df.insert(1, "Label",  y_test.values)
test_df.to_csv(OUT_TEST, index=False)

print(f"Train (after SMOTE): {len(train_df)} | Active: {(y_train_bal==1).sum()} | Inactive: {(y_train_bal==0).sum()}")
print(f"Test  (no SMOTE)  : {len(test_df)}  | Active: {(test_df['Label']==1).sum()} | Inactive: {(test_df['Label']==0).sum()}")
