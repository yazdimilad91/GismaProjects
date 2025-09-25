# ============================================
# GTEx (Heart - Left Ventricle) vs (Muscle - Skeletal)
# → Deeper Autoencoder(10) → PCA/Correlation
# → Decoder Triggering (Core = Top-250 ∩ Top-5%)
# → Multi-Enrichment (KEGG/GO/Reactome/DisGeNET/ChEA/DSigDB)
# Saves/Loads model; Train/Test split; Colab-ready
# ============================================

# 0) Install deps
!pip -q install "pandas>=2" "numpy>=1.26" "scikit-learn>=1.3" \
                tensorflow==2.* matplotlib seaborn gseapy mygene

# 1) Imports & setup
import os, gzip, urllib.request, warnings, re, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
np.random.seed(0)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

import gseapy as gp
import mygene

plt.rcParams["figure.dpi"] = 150

# 2) Download UCSC Xena Toil (GTEx/TCGA/TARGET matrix + phenotype)
expr_gz_url  = "https://toil.xenahubs.net/download/TcgaTargetGtex_rsem_gene_tpm.gz"
pheno_gz_url = "https://toil.xenahubs.net/download/TcgaTargetGTEX_phenotype.txt.gz"
expr_gz  = "TcgaTargetGtex_rsem_gene_tpm.gz"
pheno_gz = "TcgaTargetGTEX_phenotype.txt.gz"
expr_path  = "TcgaTargetGtex_rsem_gene_tpm"
pheno_path = "TcgaTargetGTEX_phenotype.txt"

def download_if_needed(url, gz, out):
    if not os.path.exists(out):
        print(f"↓ downloading {url}")
        urllib.request.urlretrieve(url, gz)
        with gzip.open(gz, "rb") as f_in, open(out, "wb") as f_out:
            f_out.write(f_in.read())
        print(f"✓ saved {out}")

download_if_needed(expr_gz_url,  expr_gz,  expr_path)
download_if_needed(pheno_gz_url, pheno_gz, pheno_path)

# 3) Phenotype (choose two normal GTEx tissues explicitly)
def read_pheno(path):
    try:
        return pd.read_csv(path, sep="\t", low_memory=False, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep="\t", low_memory=False, encoding="latin-1")

meta = read_pheno(pheno_path)

def first_present(cols, cands):
    for c in cands:
        if c in cols:
            return c
    return None

col_sample = first_present(meta.columns, ["sample","Sample","sampleID","X_sample","SAMPLE","Unnamed: 0"])
col_sample_type = first_present(meta.columns, ["_sample_type","sample_type","Sample_Type","sample type","sample.type"])
# prioritize GTEx detailed tissue labels
col_tissue = first_present(meta.columns, ["SMTSD","SMTS","tissue","_primary_site","primary_site","Primary Site","primary site","primary disease or tissue"])

if col_sample is None or col_tissue is None:
    raise ValueError("Could not locate required phenotype columns (sample / tissue). Inspect meta.columns.")

meta_small = meta[[col_sample, col_tissue] + ([col_sample_type] if col_sample_type else [])].copy()
rename_map = {col_sample:"sample", col_tissue:"tissue"}
if col_sample_type: rename_map[col_sample_type] = "sample_type"
meta_small.rename(columns=rename_map, inplace=True)
meta_small = meta_small.astype(str)

print("Phenotype columns used:", rename_map)

# --- Explicit tissues ---
TISSUE_A_TEXT = "Heart - Left Ventricle"
TISSUE_B_TEXT = "Muscle - Skeletal"

def mask_for_tissue(series, text):
    t = text.lower()
    toks = [w for w in re.split(r"[^a-z0-9]+", t) if w]
    s = series.str.lower().str.strip()
    m = pd.Series(True, index=s.index)
    for w in toks:
        m &= s.str.contains(re.escape(w), na=False)
    return m

not_tumor = True
if "sample_type" in meta_small.columns:
    st = meta_small["sample_type"].str.lower()
    not_tumor = ~st.str.contains("tumor", na=False)

mask_A = mask_for_tissue(meta_small["tissue"], TISSUE_A_TEXT) & not_tumor
mask_B = mask_for_tissue(meta_small["tissue"], TISSUE_B_TEXT) & not_tumor

samples_A_all = sorted(set(meta_small.loc[mask_A, "sample"]))
samples_B_all = sorted(set(meta_small.loc[mask_B, "sample"]))

# If no samples found, try alternate tissue columns automatically
if (len(samples_A_all) == 0 or len(samples_B_all) == 0):
    tried = [rename_map.get(col_tissue, col_tissue)]
    for alt in ["SMTSD","SMTS","tissue","_primary_site","primary_site","Primary Site","primary site","primary disease or tissue"]:
        if alt in meta.columns and alt not in tried:
            s_alt = meta[alt].astype(str)
            mA = mask_for_tissue(s_alt, TISSUE_A_TEXT)
            mB = mask_for_tissue(s_alt, TISSUE_B_TEXT)
            cA = sorted(set(meta.loc[mA, col_sample].astype(str)))
            cB = sorted(set(meta.loc[mB, col_sample].astype(str)))
            if len(cA) > 0 and len(cB) > 0:
                print(f"⚠️ Switching tissue column to '{alt}' (auto-found matches).")
                meta_small["tissue"] = s_alt.values
                samples_A_all, samples_B_all = cA, cB
                break

n_equal = min(len(samples_A_all), len(samples_B_all))
if n_equal == 0:
    # show a quick diagnostic to help users locate strings
    print("First 20 unique tissue labels example:")
    print(pd.Series(meta_small["tissue"].unique()).head(20).to_list())
    raise ValueError("No matched GTEx normals found for the requested tissues. Try checking tissue labels.")

samples_A = samples_A_all[:n_equal]
samples_B = samples_B_all[:n_equal]

print(f"Tissue A ({TISSUE_A_TEXT}) samples: {len(samples_A_all)} → using {len(samples_A)}")
print(f"Tissue B ({TISSUE_B_TEXT}) samples: {len(samples_B_all)} → using {len(samples_B)}")

# 4) Expression matrix (only matched IDs of A∪B)
with open(expr_path, "r") as f:
    header = f.readline().rstrip("\n").split("\t")
header_set = set(header[1:])

def map_to_header_form(samples, header_set):
    mapped = []
    for s in samples:
        if s in header_set:
            mapped.append(s)
        elif s.replace(".", "-") in header_set:
            mapped.append(s.replace(".", "-"))
        elif s.replace("-", ".") in header_set:
            mapped.append(s.replace("-", "."))
    return sorted(set(mapped))

mapped_A = map_to_header_form(samples_A, header_set)
mapped_B = map_to_header_form(samples_B, header_set)

usecols = ["gene"] + mapped_A + mapped_B
# Safety: if "gene" isn't the exact first col name in file, grab index 0 as gene column name
with open(expr_path, "r") as f:
    hdr = f.readline().rstrip("\n").split("\t")
gene_col_name = hdr[0]
usecols = [gene_col_name] + mapped_A + mapped_B

expr = pd.read_csv(expr_path, sep="\t", usecols=usecols)
expr.rename(columns={expr.columns[0]:"gene"}, inplace=True)
expr.set_index("gene", inplace=True)

if expr.shape[1] == 0:
    raise ValueError("No overlapping samples between phenotype and expression header after ID normalization.")

# 5) Minimal preprocessing (filter zero-sum, top variance, z-score)
expr = expr.loc[expr.sum(axis=1) > 0]
top_k = 5000 if expr.shape[0] > 5000 else expr.shape[0]
expr = expr.loc[expr.var(axis=1).sort_values(ascending=False).head(top_k).index]

scaler = StandardScaler(with_mean=True, with_std=True)
expr_z = pd.DataFrame(scaler.fit_transform(expr.T).T, index=expr.index, columns=expr.columns)

expr_A = expr_z.loc[:, mapped_A]
expr_B = expr_z.loc[:, mapped_B]
print("A matrix:", expr_A.shape, " | B matrix:", expr_B.shape)

if expr_A.shape[1] == 0 or expr_B.shape[1] == 0:
    raise ValueError("Aligned matrices are empty. Check sample-ID matching.")

# Save selected sample lists for reproducibility
pd.Series(mapped_A, name="sample").to_csv("samples_Heart_LeftVentricle.csv", index=False)
pd.Series(mapped_B, name="sample").to_csv("samples_Muscle_Skeletal.csv", index=False)

# 6) Autoencoder (deeper; train/test split; save-or-load)
input_dim    = expr_z.shape[0]
encoding_dim = 10
tf.random.set_seed(0)

def build_ae():
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(2048, activation="relu")(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512,  activation="relu")(x)
    x = layers.BatchNormalization()(x)
    latent = layers.Dense(encoding_dim, activation=None, name="latent")(x)
    x = layers.Dense(512,  activation="relu")(latent)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1024, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(2048, activation="relu")(x)
    out = layers.Dense(input_dim, activation="linear")(x)
    ae = models.Model(inp, out)
    ae.compile(optimizer="adam", loss="mse")
    return ae

MODEL_PATH = "autoencoder_model.h5"

if os.path.exists(MODEL_PATH):
    print("Loading existing model:", MODEL_PATH)
    ae = tf.keras.models.load_model(MODEL_PATH)
else:
    ae = build_ae()
    X_train, X_test = train_test_split(expr_z.T, test_size=0.1, random_state=0)
    es = callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    hist = ae.fit(X_train, X_train,
                  epochs=120, batch_size=32,
                  validation_data=(X_test, X_test),
                  verbose=1, callbacks=[es])

    # Save AE training curve (with test MSE line)
    test_recon = ae.predict(X_test, verbose=0)
    test_mse = float(np.mean(np.square(X_test - test_recon)))
    plt.figure()
    plt.plot(hist.history["loss"], label="train")
    plt.plot(hist.history["val_loss"], label="val")
    plt.axhline(y=test_mse, linestyle="--", label=f"test ({test_mse:.3f})")
    plt.xlabel("Epoch"); plt.ylabel("MSE"); plt.title("Autoencoder Training Curve"); plt.legend()
    plt.tight_layout(); plt.savefig("fig_ae_training_curve.png"); plt.close()

    ae.save(MODEL_PATH)
    pd.Index(X_test.index).to_series().to_csv("ae_test_samples.csv", index=False)

# also save encoder & decoder separately (for reuse)
inp = ae.input
latent_tensor = ae.get_layer("latent").output
encoder = models.Model(inp, latent_tensor)
encoder.save("encoder.h5")

# Rebuild decoder by reusing trained layers after 'latent'
start_idx = [i for i,l in enumerate(ae.layers) if l.name == "latent"][0]
decoder_input = layers.Input(shape=(encoding_dim,))
x = decoder_input
for lyr in ae.layers[start_idx+1:]:
    x = lyr(x)
decoder = models.Model(decoder_input, x)
decoder.save("decoder.h5")

# 7) Latent embeddings (samples × 10) + PCA (2D & save PC1..PC3 CSV)
latent_all = pd.DataFrame(
    encoder.predict(expr_z.T, verbose=0),
    index=expr_z.columns,
    columns=[f"node{i+1}" for i in range(encoding_dim)]
)

def safe(name):
    return re.sub(r"[^A-Za-z0-9_.-]+","_", name).strip("_").lower()

nameA = safe("Heart_LeftVentricle")
nameB = safe("Muscle_Skeletal")

latent_A = latent_all.loc[mapped_A]
latent_B = latent_all.loc[mapped_B]

# Save latent tables (nodes as rows, samples as columns)
latent_all.T.loc[:, mapped_A].to_csv(f"latent_nodes_{nameA}.csv")
latent_all.T.loc[:, mapped_B].to_csv(f"latent_nodes_{nameB}.csv")

# PCA (2D figure + 3 PCs CSV for A/B)
pca = PCA(n_components=3, random_state=0)
pcs = pca.fit_transform(latent_all.values)
pcs_df = pd.DataFrame(pcs, index=latent_all.index, columns=["PC1","PC2","PC3"])
pcs_df.loc[mapped_A].to_csv(f"pca3_{nameA}.csv")
pcs_df.loc[mapped_B].to_csv(f"pca3_{nameB}.csv")

coords_df = pcs_df[["PC1","PC2"]].copy()
coords_df["group"] = ["A" if s in mapped_A else ("B" if s in mapped_B else "Other")
                      for s in coords_df.index]
plt.figure()
sns.scatterplot(data=coords_df, x="PC1", y="PC2", hue="group", s=14)
plt.title("Latent Space (PCA, 2D)")
plt.tight_layout(); plt.savefig("fig_latent_pca.png"); plt.close()

# 8) Correlation 10×10 (A, B) + heatmaps + CSV
def corr10(df_latent):
    return df_latent.corr(method="pearson").fillna(0.0).clip(-1,1)

corr_A = corr10(latent_A)
corr_B = corr10(latent_B)

def save_heatmap(C, title, path):
    plt.figure(figsize=(6,5))
    sns.heatmap(C, cmap="vlag", center=0, annot=True, fmt=".2f")
    plt.title(title)
    plt.tight_layout(); plt.savefig(path); plt.close()

save_heatmap(corr_A, f"Correlation ({nameA})", f"fig_corr_{nameA}.png")
save_heatmap(corr_B, f"Correlation ({nameB})", f"fig_corr_{nameB}.png")
corr_A.to_csv(f"corr10x10_{nameA}.csv")
corr_B.to_csv(f"corr10x10_{nameB}.csv")

# 9) Decoder Triggering (core genes = Top-250 ∩ Top-5%)
baseline = decoder.predict(np.zeros((1, encoding_dim), dtype=np.float32), verbose=0)[0]
genes = expr_z.index.values

os.makedirs("node2genes_core", exist_ok=True)

def core_genes_triggering(decoder, node_index, gene_index, k=250, perc=95, amp=1.0):
    vec = np.zeros((1, encoding_dim), dtype=np.float32)
    vec[0, node_index] = amp
    out = decoder.predict(vec, verbose=0)[0]
    delta = out - baseline
    top_idx = np.argsort(-np.abs(delta))[:k]
    thr = np.percentile(np.abs(delta), perc)
    mask = np.where(np.abs(delta) >= thr)[0]
    core_idx = np.unique(np.concatenate([top_idx, mask]))
    return [gene_index[i] for i in core_idx]

# Gene symbol mapping
mg = mygene.MyGeneInfo()

def to_symbols(ensembl_or_gene):
    ids = [g.split(".")[0] for g in ensembl_or_gene]  # strip version
    try:
        out = mg.querymany(ids, scopes="ensembl.gene", fields="symbol",
                           species="human", as_dataframe=True, df_index=True,
                           returnall=False, verbose=False)
    except Exception as e:
        print("MyGene query failed:", e)
        # fallback → return raw ids
        return list(sorted(set(ensembl_or_gene)))
    symbols = []
    for i, g in enumerate(ids):
        try:
            sym = out.loc[g, "symbol"]
            if isinstance(sym, pd.Series):
                sym = sym.iloc[0]
            if pd.isna(sym):
                sym = ensembl_or_gene[i]
        except Exception:
            sym = ensembl_or_gene[i]
        symbols.append(sym)
    return list(sorted(set(symbols)))

# Multi-enrichment helper
gene_sets = [
    'KEGG_2021_Human',
    'GO_Biological_Process_2021',
    'GO_Molecular_Function_2021',
    'GO_Cellular_Component_2021',
    'Reactome_2022',
    'DisGeNET',
    'ChEA_2022',
    'DSigDB'
]

def run_enrichment(symbols, node_name):
    if len(symbols) < 5:
        return pd.DataFrame([{"Gene_set":"", "Term":"", "Adjusted P-value":"", "Overlap":"",
                              "Genes":"", "Node":node_name, "Note":"<5 genes; enrichment skipped>"}])
    try:
        enr = gp.enrichr(gene_list=symbols, gene_sets=gene_sets,
                         organism='Human', outdir=None, cutoff=0.05)
        res = enr.results.copy()
        res["Node"] = node_name
        return res
    except Exception as e:
        print(f"Enrichr failed for {node_name}:", e)
        return pd.DataFrame([{"Gene_set":"", "Term":"", "Adjusted P-value":"", "Overlap":"",
                              "Genes":"", "Node":node_name, "Note":"enrichr failed"}])

for j in range(encoding_dim):
    node_name = f"node{j+1}"
    core = core_genes_triggering(decoder, j, genes, k=250, perc=95, amp=1.0)
    symbols = to_symbols(core)
    pd.Series(core, name="gene_id").to_csv(f"node2genes_core/{node_name}_core_ids.csv", index=False)
    pd.Series(symbols, name="symbol").to_csv(f"node2genes_core/{node_name}_core_symbols.csv", index=False)
    enr = run_enrichment(symbols, node_name)
    enr.to_csv(f"node2genes_core/{node_name}_enrichment.csv", index=False)

print("All outputs saved:")
print([
 "autoencoder_model.h5","encoder.h5","decoder.h5",
 "fig_ae_training_curve.png","fig_latent_pca.png",
 f"fig_corr_{nameA}.png",f"fig_corr_{nameB}.png",
 f"corr10x10_{nameA}.csv",f"corr10x10_{nameB}.csv",
 f"latent_nodes_{nameA}.csv",f"latent_nodes_{nameB}.csv",
 f"pca3_{nameA}.csv",f"pca3_{nameB}.csv",
 "node2genes_core/*.csv"
])
