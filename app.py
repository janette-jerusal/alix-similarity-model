# app.py
# Streamlit app: User Story Similarity with selectable output columns for Story A and Story B

import io
import re
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="User Story Similarity", layout="wide")
st.title("User Story Similarity (TF-IDF + Cosine)")
st.caption("Upload two Excel files, map columns, choose which fields to include in the final output.")


# ----------------------------
# Helpers
# ----------------------------
def normalize_colname(x: str) -> str:
    return re.sub(r"\s+", " ", str(x)).strip()

def read_excel_any(uploaded_file) -> dict:
    """Return dict of {sheet_name: df} with normalized column names."""
    xls = pd.ExcelFile(uploaded_file)
    out = {}
    for sh in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sh)
        df.columns = [normalize_colname(c) for c in df.columns]
        out[sh] = df
    return out

def safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)

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
):
    # Text arrays
    a_text = df_a[a_desc_col].fillna("").astype(str).tolist()
    b_text = df_b[b_desc_col].fillna("").astype(str).tolist()

    # If either side empty, return empty
    if len(a_text) == 0 or len(b_text) == 0:
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

    sims = cosine_similarity(Xa, Xb)  # shape: [len(A), len(B)]

    # Build pair table efficiently
    rows = []
    for i in range(sims.shape[0]):
        row = sims[i]
        if top_k is not None and top_k > 0:
            # take top_k indices then filter by threshold
            idx = np.argpartition(-row, min(top_k, len(row)) - 1)[: min(top_k, len(row))]
            idx = idx[np.argsort(-row[idx])]
        else:
            # all
            idx = np.arange(len(row))

        for j in idx:
            score = float(row[j])
            if score >= threshold:
                rows.append((i, j, score))

    if not rows:
        return pd.DataFrame()

    pairs = pd.DataFrame(rows, columns=["_ai", "_bj", "similarity"])

    # Pull ids
    pairs["A_id"] = df_a.iloc[pairs["_ai"]][a_id_col].astype(str).values
    pairs["B_id"] = df_b.iloc[pairs["_bj"]][b_id_col].astype(str).values

    # Attach selected metadata (prefixed)
    # Always include descriptions in output if selected by user; you control via keep cols lists
    a_meta = df_a[[a_id_col] + [c for c in a_keep_cols if c != a_id_col]].copy()
    b_meta = df_b[[b_id_col] + [c for c in b_keep_cols if c != b_id_col]].copy()

    # Rename to A_/B_
    a_meta = a_meta.rename(columns={c: ("A_id" if c == a_id_col else f"A_{c}") for c in a_meta.columns})
    b_meta = b_meta.rename(columns={c: ("B_id" if c == b_id_col else f"B_{c}") for c in b_meta.columns})

    # Merge
    out = pairs.merge(a_meta, on="A_id", how="left").merge(b_meta, on="B_id", how="left")

    # Clean helper cols
    out = out.drop(columns=["_ai", "_bj"])

    # Order columns nicely
    base = ["similarity", "A_id", "B_id"]
    a_cols = [c for c in out.columns if c.startswith("A_")]
    b_cols = [c for c in out.columns if c.startswith("B_")]
    ordered = base + sorted(a_cols) + sorted(b_cols)
    ordered = [c for c in ordered if c in out.columns]
    out = out[ordered].sort_values("similarity", ascending=False).reset_index(drop=True)

    return out


def to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


# ----------------------------
# UI: Uploads
# ----------------------------
col1, col2 = st.columns(2)

with col1:
    file_a = st.file_uploader("Upload Excel for Story A", type=["xlsx", "xls"], key="file_a")
with col2:
    file_b = st.file_uploader("Upload Excel for Story B", type=["xlsx", "xls"], key="file_b")

if not file_a or not file_b:
    st.info("Upload both files to continue.")
    st.stop()

sheets_a = read_excel_any(file_a)
sheets_b = read_excel_any(file_b)

# ----------------------------
# UI: Sheet selection
# ----------------------------
col1, col2 = st.columns(2)
with col1:
    sheet_a = st.selectbox("Sheet for Story A", list(sheets_a.keys()))
with col2:
    sheet_b = st.selectbox("Sheet for Story B", list(sheets_b.keys()))

df_a = sheets_a[sheet_a]
df_b = sheets_b[sheet_b]

if df_a.empty or df_b.empty:
    st.warning("One of the selected sheets is empty.")
    st.stop()

cols_a = list(df_a.columns)
cols_b = list(df_b.columns)

# ----------------------------
# UI: Column mapping
# ----------------------------
st.subheader("Column mapping")

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

# Ensure description is present if user wants it; you can force it:
if a_desc_col not in a_keep and st.checkbox("Force include A description in output", value=True):
    a_keep = list(dict.fromkeys(a_keep + [a_desc_col]))
if b_desc_col not in b_keep and st.checkbox("Force include B description in output", value=True):
    b_keep = list(dict.fromkeys(b_keep + [b_desc_col]))

# ----------------------------
# UI: Similarity controls
# ----------------------------
st.subheader("Similarity settings")

c1, c2, c3, c4 = st.columns(4)
with c1:
    threshold = st.slider("Threshold", 0.0, 1.0, 0.50, 0.01)
with c2:
    top_k = st.number_input("Top-K matches per A (0 = all)", min_value=0, max_value=5000, value=20, step=5)
with c3:
    ngram_max = st.selectbox("Max n-gram", [1, 2, 3], index=1)  # default 2
with c4:
    max_features = st.number_input("Max features (0 = unlimited)", min_value=0, max_value=200000, value=50000, step=5000)

top_k_val = None if top_k == 0 else int(top_k)
max_features_val = None if max_features == 0 else int(max_features)

# ----------------------------
# Run
# ----------------------------
if st.button("Compare", type="primary"):
    with st.spinner("Computing similarity..."):
        # always include ID in keep list
        a_keep_cols = [a_id_col] + [c for c in a_keep if c in df_a.columns]
        b_keep_cols = [b_id_col] + [c for c in b_keep if c in df_b.columns]

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
            top_k=top_k_val,
            ngram_max=int(ngram_max),
            max_features=max_features_val,
        )

    if results_df.empty:
        st.warning("No matches found at this threshold.")
        st.stop()

    st.success(f"Found {len(results_df):,} matching pairs.")
    st.dataframe(results_df, use_container_width=True)

    # Downloads
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
