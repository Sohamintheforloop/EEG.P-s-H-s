import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 0) Paths & output locations
# ----------------------------
path = r"C:\Users\Soham\GSE33000_raw_data.txt"  # input TSV (tab-separated values)
in_path = Path(path)
out_dir = in_path.parent
stem = in_path.stem  # e.g., "GSE33000_raw_data"

expr_out_path = out_dir / f"{stem}_processed_expression.csv"
labels_out_path = out_dir / f"{stem}_labels.csv"

print("Loading...")
df = pd.read_csv(path, sep="\t", comment="!", low_memory=False)

# ---------------------------------------------------------
# 1) Try to set gene/probe IDs as index if present
# ---------------------------------------------------------
for id_col in ["ID_REF", "ProbeID", "ID", "gene_id"]:
    if id_col in df.columns:
        df = df.set_index(id_col)
        break

# ---------------------------------------------------------
# 2) Convert all cell values to numeric (others -> NaN)
# ---------------------------------------------------------
df_num = df.apply(pd.to_numeric, errors="coerce")

# ---------------------------------------------------------
# 3) Remove columns that are fully NaN; then rows fully NaN
# ---------------------------------------------------------
df_num = df_num.dropna(axis=1, how="all").dropna(axis=0, how="all")

# ---------------------------------------------------------
# 4) Per-gene (row-wise) mean imputation across samples
#    (more biologically sensible than per-sample)
# ---------------------------------------------------------
df_num = df_num.T.fillna(df_num.T.mean()).T

# ---------------------------------------------------------
# 5) Log2 check — robust (use quantiles + cap)
# ---------------------------------------------------------
q95 = df_num.stack().quantile(0.95)
vmax = float(df_num.max().max())
already_log = (q95 <= 20) and (vmax <= 100)

if already_log:
    expr = df_num
else:
    expr = np.log2(df_num + 1)

# ---------------------------------------------------------
# 6) Remove (near) zero-variance genes
# ---------------------------------------------------------
gene_var = expr.var(axis=1)
expr = expr.loc[gene_var > 1e-8]

# ---------------------------------------------------------
# 7) Transpose → samples × genes, ensure finite
# ---------------------------------------------------------
X_df = expr.T.copy()  # rows = samples, cols = genes
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(X_df.mean())

sample_ids = X_df.index.astype(str).tolist()

# ---------------------------------------------------------
# 8) Standardize (z-score across samples for each gene)
# ---------------------------------------------------------
scaler = StandardScaler(with_mean=True, with_std=True)
X_scaled = scaler.fit_transform(X_df.values)

# Keep as DataFrame to preserve labels/columns
X_scaled_df = pd.DataFrame(X_scaled, index=sample_ids, columns=X_df.columns)

print("Preprocessing completed")
print("Final matrix shape (samples × genes):", X_scaled_df.shape)

# ---------------------------------------------------------
# 9) Parse labels (HD vs control) from a GEO series matrix
# ---------------------------------------------------------
def parse_labels_from_series_matrix(series_matrix_path, sample_ids):
    """
    Parse labels (HD vs control) from a GEO series matrix file.
    Returns a list y aligned to sample_ids order (1=HD, 0=control, None=unknown).
    """
    meta_lines = []
    with open(series_matrix_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("!"):
                meta_lines.append(line.rstrip("\n"))

    gsm_pattern  = re.compile(r"^!Sample_geo_accession = (GSM\d+)$")
    title_pattern = re.compile(r"^!Sample_title = (.+)$")
    # Be permissive: handles ch1/ch2 and numbered fields
    char_pattern = re.compile(r"^!Sample_characteristics.* = (.+)$", re.IGNORECASE)

    gsm_order = []
    characteristics = {}  # gsm -> list[str]
    titles = {}

    current_gsm_idx = -1
    for ln in meta_lines:
        m_gsm = gsm_pattern.match(ln)
        if m_gsm:
            current_gsm_idx += 1
            gsm = m_gsm.group(1)
            gsm_order.append(gsm)
            characteristics[gsm] = []
            continue

        m_title = title_pattern.match(ln)
        if m_title and 0 <= current_gsm_idx < len(gsm_order):
            titles[gsm_order[current_gsm_idx]] = m_title.group(1)
            continue

        m_char = char_pattern.match(ln)
        if m_char and 0 <= current_gsm_idx < len(gsm_order):
            characteristics[gsm_order[current_gsm_idx]].append(m_char.group(1))
            continue

    def label_from_text(txt):
        t = txt.lower()
        if any(k in t for k in ["huntington", "hd", "patient", "case", "disease"]):
            return 1
        if any(k in t for k in ["control", "healthy", "normal"]):
            return 0
        return None

    disease = {}
    for gsm in gsm_order:
        lbl = None
        for c in characteristics.get(gsm, []):
            parts = c.split(":")
            cand = parts[-1].strip() if len(parts) > 1 else c.strip()
            lbl = label_from_text(cand)
            if lbl is not None:
                break
            lbl = label_from_text(c)
            if lbl is not None:
                break
        if lbl is None and gsm in titles:
            lbl = label_from_text(titles[gsm])
        if lbl is not None:
            disease[gsm] = lbl

    def normalize_id(s):
        m = re.search(r"(GSM\d+)", s)
        return m.group(1) if m else s

    y = []
    unresolved = []
    for s in sample_ids:
        gsm = normalize_id(s)
        lbl = disease.get(gsm, label_from_text(s))
        y.append(lbl)
        if lbl is None:
            unresolved.append(s)

    return y, unresolved

series_matrix_path = r"C:\Users\Soham\GSE33000_series_matrix.txt"
y = None

try:
    y_list, unresolved = parse_labels_from_series_matrix(series_matrix_path, sample_ids)
    if unresolved:
        print(f"⚠ Could not infer labels for {len(unresolved)} samples (e.g., {unresolved[:3]}). "
              f"Defaulting to control (0) for now — please review {labels_out_path.name} later.")
    # Default unresolved -> control
    y = np.asarray([0 if v is None else v for v in y_list], dtype=int)
    print("Parsed labels from series matrix. Shape:", y.shape)
except FileNotFoundError:
    print("⚠ Series matrix not found at:", series_matrix_path)
except Exception as e:
    print("⚠ Could not parse labels from series matrix:", e)

# Fallback: if still None, create control-only labels (edit later if needed)
if y is None:
    y = np.zeros(len(sample_ids), dtype=int)
    print("✅ Using default labels (all control=0). Update the labels CSV later if needed.")

assert len(y) == X_scaled_df.shape[0], f"Label count {len(y)} != samples {X_scaled_df.shape[0]}"

# ---------------------------------------------------------
# 10) SAVE: processed matrix and labels next to input file
# ---------------------------------------------------------
try:
    X_scaled_df.to_csv(expr_out_path)
    pd.DataFrame({"sample_id": sample_ids, "label": y}).to_csv(labels_out_path, index=False)
    print("Saved processed expression to:", expr_out_path)
    print("Saved labels to:", labels_out_path)
except Exception as e:
    print("❌ Error saving CSV files:", e)

print("Done.")
