# app.py
# Streamlit app: User Story Similarity with 1-file (self-compare) OR 2-file (A vs B) mode
# Fixes duplicate column name crashes by making headers unique + deduping selected columns
# Output includes selectable A_/B_ fields (Topic/Status/Disposition/ID/Description etc.)

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="User Story Similarity", layout="wide")
st.title("User Story Similarity (TF-IDF + Cosine)")
st.caption("Upload one Excel file (self-compare) or two Excel files (cross-compare). Choose which fields appear in the output.")


# ----------------------------
# Helpers
# ----------------------------
def normalize_colname(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()

def dedupe_preserve_order(cols):
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def make_unique_columns(cols):
    """
    Make column names unique by appending .2, .3, ...
    Streamlit/pyarrow errors if df has duplicate column names.
    """
    counts = {}
    out = []
    for c in cols:
        c = str(c)
        if c not in counts:
            counts[c] = 1
            out.append(c)
        else:
            counts[c] += 1
            out.append(f"{c}.{counts[c]}")
    return out

def read_excel_any(uploaded_file) -> dict:
    """Return dict of {sheet_name: df} with normalized + UNIQUE column names."""
    xls = pd.ExcelFile(uploaded_file)
    out = {}
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh)

        # normalize then make unique
        cols = [normalize_colname(c) for c in df.columns]
        df.columns = make_unique_columns(cols)

        out[sh] = df
    return out

def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()

def build_results(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    a_id_col: str,
    a_desc_col: str,
    b_id_col: str,
    b_desc_col: str,
    a_keep_cols: list[str],
    b_keep_cols: list[str],
    threshold: float,
    top_k: int | None,
    ngram_max: int,
    max_features: int | None,
    self_mode: bool,
    dedupe_symmetric: bool,
):
    # Text arrays
    a_text = df_a[a_desc_col].fillna("").astype(str).tolist()
    b_text = df_b[b_desc_col].fillna("").astype(str).tolist()
    if not a_text or not b_text:
        return pd.DataFrame()

    # Fit vectorizer on combined corpus to share vocab
    corpus = a_text + b_text
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, ngram_max),
        max_features=max_features,
    )
    X = vectorizer.fit_transform(corpus)
    Xa = X[: len(a_text)]
    Xb = X[len(a_text) :]

    sims = cosine_similarity(Xa, Xb)  # [len(A), len(B)]

    # Build pair list
    rows = []
    for i in range(sims.shape[0]):
        row = sims[i]

        if top_k is not None and top_k > 0:
            k = min(top_k, len(row))
            idx = np.argpartition(-row, k - 1)[:k]
            idx = idx[np.argsort(-row[idx])]
        else:
            idx = np.arange(len(row))

        for j in idx:
            if self_mode and i == j:
                continue  # don't match a story to itself
            if self_mode and dedupe_symmetric and j < i:
                continue  # avoid mirror duplicates (keep only i < j)
            score = float(row[j])
            if score >= threshold:
                rows.append((i, j, score))

    if not rows:
        return pd.DataFrame()

    pairs = pd.DataFrame(rows, columns=["_ai", "_bj", "similarity"])

    # IDs
    pairs["A_id"] = df_a.iloc[pairs["_ai"]][a_id_col].astype(str).values
    pairs["B_id"] = df_b.iloc[pairs["_bj"]][b_id_col].astype(str).values

    # Metadata (dedup keep cols again just in case)
    a_keep_cols = dedupe_preserve_order([c for c in a_keep_cols if c in df_a.columns])
    b_keep_cols = dedupe_preserve_order([c for c in b_keep_cols if c in df_b.columns])

    a_meta = df_a[a_keep_cols].copy()
    b_meta = df_b[b_keep_cols].copy()

    # Rename to A_/B_
    a_meta = a_meta.rename(columns={c: ("A_id" if c == a_id_col else f"A_{c}") for c in a_meta.columns})
    b_meta = b_meta.rename(columns={c: ("B_id" if c == b_id_col else f"B_{c}") for c in b_meta.columns})

    # Merge metadata onto pairs
    out = pairs.merge(a_meta, on="A_id", how="left").merge(b_meta, on="B_id", how="left")
    out = out.drop(columns=["_ai", "_bj"])

    # Order columns: similarity, A..., B...
    base = ["similarity", "A_id", "B_id"]
    a_cols = [c for c in out.columns if c.startswith("A_")]
    b_cols = [c for c in out.columns if c.startswith("B_")]
    ordered = base + sorted(a_cols) + sorted(b_cols)
    ordered = [c for c in ordered if c in out.columns]
    out = out[ordered].sort_values("similarity", ascending=False).reset_index(drop=True)

    # Final safety: make output headers unique for Streamlit/pyarrow
    out.columns = make_unique_columns(out.columns)

    return out


# ----------------------------
# Uploads (File B optional)
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    file_a = st.file_uploader("Upload Excel (Story A) — required", type=["xlsx", "xls"], key="file_a")
with col2:
    file_b = st.file_uploader("Upload Excel (Story B) — optional", type=["xlsx", "xls"], key="file_b")

if not file_a:
    st.info("Upload at least one Excel file to continue.")
    st.stop()

sheets_a = read_excel_any(file_a)
sheets_b = read_excel_any(file_b) if file_b else None

# Mode
self_mode = file_b is None
st.subheader("Mode")
st.write("**Self-compare mode** (1 file) ✅" if self_mode else "**Cross-compare mode** (2 files) ✅")

# ----------------------------
# Sheet selection
# ----------------------------
c1, c2 = st.columns(2)
with c1:
    sheet_a = st.selectbox("Sheet for Story A", list(sheets_a.keys()))

with c2:
    if sheets_b:
        sheet_b = st.selectbox("Sheet for Story B", list(sheets_b.keys()))
    else:
        sheet_b = sheet_a  # same sheet

df_a = sheets_a[sheet_a]
df_b = sheets_b[sheet_b] if sheets_b else df_a.copy()

if df_a.empty or df_b.empty:
    st.warning("One of the selected sheets is empty.")
    st.stop()

# ----------------------------
# Column mapping
# ----------------------------
st.subheader("Column mapping")

cols_a = list(df_a.columns)
cols_b = list(df_b.columns)

m1, m2 = st.columns(2)
with m1:
    st.markdown("**Story A**")
    a_id_col = st.selectbox("A: ID column", cols_a, index=cols_a.index("ID") if "ID" in cols_a else 0)
    a_desc_col = st.selectbox(
        "A: Description column",
        cols_a,
        index=cols_a.index("Description") if "Description" in cols_a else min(1, len(cols_a) - 1),
    )
    a_optional = [c for c in cols_a if c not in {a_id_col}]
    a_defaults = [c for c in ["Topic", "Status", "Disposition", "Description"] if c in cols_a and c != a_id_col]
    a_keep = st.multiselect(
        "A: fields to include in output (besides ID)",
        options=a_optional,
        default=a_defaults if a_defaults else a_optional,
    )

with m2:
    st.markdown("**Story B**")
    b_id_col = st.selectbox("B: ID column", cols_b, index=cols_b.index("ID") if "ID" in cols_b else 0)
    b_desc_col = st.selectbox(
        "B: Description column",
        cols_b,
        index=cols_b.index("Description") if "Description" in cols_b else min(1, len(cols_b) - 1),
    )
    b_optional = [c for c in cols_b if c not in {b_id_col}]
    b_defaults = [c for c in ["Topic", "Status", "Disposition", "Description"] if c in cols_b and c != b_id_col]
    b_keep = st.multiselect(
        "B: fields to include in output (besides ID)",
        options=b_optional,
        default=b_defaults if b_defaults else b_optional,
    )

# Force include descriptions (recommended)
force_a_desc = st.checkbox("Force include A description in output", value=True)
force_b_desc = st.checkbox("Force include B description in output", value=True)

if force_a_desc and a_desc_col not in a_keep:
    a_keep = list(dict.fromkeys(a_keep + [a_desc_col]))
if force_b_desc and b_desc_col not in b_keep:
    b_keep = list(dict.fromkeys(b_keep + [b_desc_col]))

# Build keep col lists (always include ID) + dedupe
a_keep_cols = dedupe_preserve_order([a_id_col] + [c for c in a_keep if c in df_a.columns])
b_keep_cols = dedupe_preserve_order([b_id_col] + [c for c in b_keep if c in df_b.columns])

# ----------------------------
# Similarity settings
# ----------------------------
st.subheader("Similarity settings")

s1, s2, s3, s4 = st.columns(4)
with s1:
    threshold = st.slider("Minimum similarity score (0–1)", 0.0, 1.0, 0.60, 0.01)
with s2:
    top_k_in = st.number_input("Top-K matches per A (0 = all)", min_value=0, max_value=5000, value=20, step=5)
with s3:
    ngram_max = st.selectbox("Max n-gram", [1, 2, 3], index=1)
with s4:
    max_features_in = st.number_input("Max features (0 = unlimited)", min_value=0, max_value=200000, value=50000, step=5000)

top_k = None if top_k_in == 0 else int(top_k_in)
max_features = None if max_features_in == 0 else int(max_features_in)

dedupe_symmetric = False
if self_mode:
    dedupe_symmetric = st.checkbox("In self-compare, remove mirror duplicates (keep only one of A↔B)", value=True)

# ----------------------------
# Run
# ----------------------------
if st.button("Compare", type="primary"):
    with st.spinner("Computing similarity..."):
        results_df = build_results(
            df_a=df_a,
            df_b=df_b,
            a_id_col=a_id_col,
            a_desc_col=a_desc_col,
            b_id_col=b_id_col,
            b_desc_col=b_desc_col,
            a_keep_cols=a_keep_cols,
            b_keep_cols=b_keep_cols,
            threshold=float(threshold),
            top_k=top_k,
            ngram_max=int(ngram_max),
            max_features=max_features,
            self_mode=self_mode,
            dedupe_symmetric=dedupe_symmetric,
        )

    if results_df.empty:
        st.warning("No matches found at this threshold.")
        st.stop()

    st.success(f"Found {len(results_df):,} matching pairs.")
    st.dataframe(results_df, use_container_width=True)

    st.download_button(
        "Download CSV",
        data=results_df.to_csv(index=False).encode("utf-8"),
        file_name="user_story_similarity_results.csv",
        mime="text/csv",
    )

    st.download_button(
        "Download Excel",
        data=to_excel_bytes(results_df),
        file_name="user_story_similarity_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

