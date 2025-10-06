import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="LGBM Classifier", page_icon="✅", layout="wide")
st.title("LGBM Classifier – Simple Deployment")

@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

artifact = load_artifact("lgbm_artifact.joblib")
model = artifact["model"]
train_cols = artifact["columns"]

st.write("Upload a CSV with the **same feature columns** used in training.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Keep only training columns (extra columns dropped, missing ones filled)
    X = df.reindex(columns=train_cols)
    # Simple handling for missing numeric/categorical values
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = X[c].fillna(0)
        else:
            X[c] = X[c].fillna("")

    # Predict
    preds = model.predict(X)
    proba = None
    try:
        proba = model.predict_proba(X)
    except Exception:
        pass

    st.subheader("Predictions")
    out = pd.DataFrame({"prediction": preds})
    if proba is not None and proba.ndim == 2:
        proba_df = pd.DataFrame(proba, columns=[f"prob_{i}" for i in range(proba.shape[1])])
        out = pd.concat([out, proba_df], axis=1)

    st.dataframe(out, use_container_width=True)

    # Optional: download predictions
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download predictions.csv", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
else:
    st.info("Waiting for file...")

st.caption("Click Here")
