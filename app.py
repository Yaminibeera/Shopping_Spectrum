from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent

SEGMENT_DESCRIPTIONS = {
    "High-Value": "Recent, frequent, and big spenders. Prioritize for loyalty programs.",
    "Regular": "Steady purchasers but not premium. Good candidates for upselling.",
    "Occasional": "Rare, occasional purchases. Consider re-engagement campaigns.",
    "At-Risk": "Haven't purchased in a long time. Target with win-back offers.",
}


@st.cache_resource
def load_models():
    with open(BASE_DIR / "kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "cluster_labels.pkl", "rb") as f:
        cluster_labels = pickle.load(f)
    with open(BASE_DIR / "top_similar_products.pkl", "rb") as f:
        similar_products = pickle.load(f)
    return kmeans, scaler, cluster_labels, similar_products


def find_product_match(query: str, product_names: list[str]) -> str | None:
    query = query.strip().upper()
    if not query:
        return None

    for name in product_names:
        if name.upper() == query:
            return name

    partial_matches = [name for name in product_names if query in name.upper()]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if partial_matches:
        return partial_matches[0]
    return None


st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="wide")

st.markdown(
    """
    <style>
    .recommendation-card {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
        border: 1px solid #dbeafe;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
    }
    .segment-badge {
        display: inline-block;
        padding: 0.35rem 0.9rem;
        border-radius: 999px;
        font-weight: 700;
        margin-bottom: 0.75rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🛒 Shopper Spectrum")
st.caption("Customer Segmentation & Product Recommendation System")

required_files = [
    BASE_DIR / "kmeans_model.pkl",
    BASE_DIR / "scaler.pkl",
    BASE_DIR / "cluster_labels.pkl",
    BASE_DIR / "top_similar_products.pkl",
]

if not all(path.exists() for path in required_files):
    st.error(
        "Model files are missing. Run `python train_models.py` first, "
        "then restart this app."
    )
    st.stop()

kmeans, scaler, cluster_labels, similar_products = load_models()
product_names = sorted(similar_products.keys())

tab_reco, tab_segment = st.tabs(["Product Recommendation", "Customer Segmentation"])

with tab_reco:
    st.header("🎁 Product Recommendation")
    st.write("Enter a product name to get 5 similar product suggestions.")

    product_input = st.text_input("Product Name", placeholder="e.g. WHITE METAL LANTERN")

    if st.button("Get Recommendations", type="primary"):
        matched_product = find_product_match(product_input, product_names)

        if not product_input.strip():
            st.warning("Please enter a product name.")
        elif matched_product is None:
            st.error("Product not found. Try another name or a partial match.")
        else:
            recommendations = similar_products[matched_product][:5]
            st.success(f"Top 5 products similar to: **{matched_product}**")

            for i, (prod, score) in enumerate(recommendations, 1):
                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <strong>#{i}</strong> {prod}<br>
                        <span style="color:#475569;">Similarity score: {score:.3f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

with tab_segment:
    st.header("🎯 Customer Segmentation")
    st.write("Enter customer purchase behavior to predict their segment.")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    with col2:
        frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    with col3:
        monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0, format="%.2f")

    if st.button("Predict Cluster", type="primary"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = int(kmeans.predict(input_scaled)[0])
        segment = cluster_labels.get(cluster, "Unknown")

        colors = {
            "High-Value": "#166534",
            "Regular": "#1d4ed8",
            "Occasional": "#c2410c",
            "At-Risk": "#b91c1c",
        }
        badge_color = colors.get(segment, "#334155")

        st.markdown(
            f"""
            <div class="segment-badge" style="background:{badge_color}22;color:{badge_color};">
                {segment}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(SEGMENT_DESCRIPTIONS.get(segment, "Segment profile unavailable."))

        st.dataframe(
            pd.DataFrame(
                {
                    "Metric": ["Recency (days)", "Frequency", "Monetary (£)"],
                    "Input Value": [recency, frequency, monetary],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
