import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams["svg.fonttype"] = "none"
os.makedirs("results", exist_ok=True)

PALETTE = {0: "steelblue", 1: "forestgreen"}
LABELS  = {0: "Inactive (Decoy)", 1: "Active"}

for split in ["train", "test"]:
    df = pd.read_csv(f"{split}_data.csv")

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, grp in df.groupby("Label"):
        ax.scatter(grp["MolWt"], grp["MolLogP"],
                   c=PALETTE[label], label=LABELS[label],
                   alpha=0.65, s=30, edgecolors="none")

    ax.set_xlabel("Molecular Weight (Da)", fontsize=12)
    ax.set_ylabel("LogP", fontsize=12)
    ax.set_title(f"Chemical Space — {split.capitalize()} Set", fontsize=13, fontweight="bold")
    ax.legend(title="Class", fontsize=10)
    ax.grid(False)
    plt.tight_layout()
    plt.savefig(f"{split}.svg")
    plt.savefig(f"{split}.tiff", dpi=600)
    plt.close()
    print(f"Saved: Chemical_space_{split}.tiff")
