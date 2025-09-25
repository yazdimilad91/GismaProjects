import urllib.request, gzip, os
import pandas as pd
from sklearn.preprocessing import StandardScaler


# --- Download expression ---
urllib.request.urlretrieve(
    "https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz",
    "expr.gz"
)
with gzip.open("expr.gz","rb") as f_in, open("expr.txt","wb") as f_out:
    f_out.write(f_in.read())

# --- Download phenotype ---
urllib.request.urlretrieve(
    "https://toil.xenahubs.net/download/TcgaTargetGTEX_phenotype.txt.gz",
    "pheno.gz"
)
with gzip.open("pheno.gz","rb") as f_in, open("pheno.txt","wb") as f_out:
    f_out.write(f_in.read())

# --- Read phenotype and filter TCGA breast samples ---
pheno = pd.read_csv("pheno.txt", sep="\t", low_memory=False, encoding="latin-1")
pheno = pheno.rename(columns=str.lower)

breast = pheno[pheno["primary disease or tissue"].str.contains("breast", case=False, na=False)]
samples = set(breast["sample"])

# --- Read expression and keep matched samples ---
with open("expr.txt") as f:
    header = f.readline().strip().split("\t")

usecols = [0] + [i for i,s in enumerate(header) if s in samples or s.replace(".","-") in samples]
expr = pd.read_csv("expr.txt", sep="\t", usecols=usecols)
expr = expr.rename(columns={expr.columns[0]:"gene"}).set_index("gene")

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from combat.pycombat import pycombat
from sklearn.decomposition import PCA
import matplotlib.patches as mpatches

# align meta to expr
breast2 = breast.copy()
breast2["sample"] = breast2["sample"].astype(str).str.replace(".","-", regex=False)
meta = breast2.set_index("sample").reindex(expr.columns)

# map sample types
label_map = {"Normal Tissue":"Control", "Solid Tissue Normal":"Control",
             "Primary Tumor":"Tumor", "Metastatic":"Tumor"}
meta["status"] = meta["_sample_type"].map(label_map)

# source (TCGA vs GTEx)
meta["source"] = np.where(meta.index.str.startswith("TCGA"), "TCGA", "GTEx")

# combine source + status
meta["group"] = meta["source"] + "_" + meta["status"]

# ComBat
batch = pd.Series(meta["source"].values, index=expr.columns)
X_corrected = pycombat(expr, batch)
X_corrected = pd.DataFrame(X_corrected, index=expr.index, columns=expr.columns)

# color map with clear names
color_map = {
    "TCGA_Control":"green",
    "TCGA_Tumor":"red",
    "GTEx_Control":"blue",
    "GTEx_Tumor":"orange"
}
colors = meta["group"].map(color_map).fillna("gray").values

# PCA
pca_raw = PCA(n_components=2).fit_transform(expr.T)
pca_corr = PCA(n_components=2).fit_transform(X_corrected.T)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(pca_raw[:,0], pca_raw[:,1], c=colors)
plt.title("Before ComBat"); plt.xlabel("PC1"); plt.ylabel("PC2")

plt.subplot(1,2,2)
plt.scatter(pca_corr[:,0], pca_corr[:,1], c=colors)
plt.title("After ComBat"); plt.xlabel("PC1"); plt.ylabel("PC2")

handles = [mpatches.Patch(color=color_map[k], label=k) for k in color_map]
plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


X_corrected = X_corrected.loc[X_corrected.var(1).nlargest(500).index]

# --- Scale data (samples × genes) ---
X = StandardScaler().fit_transform(X_corrected.T)
X = pd.DataFrame(X, index=X_corrected.columns, columns=X_corrected.index)

# --- Save results ---
os.makedirs("prepared", exist_ok=True)
X.to_csv("prepared/X_tcga.csv")
