import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ===============================
# Load saved models
# ===============================
@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, "kmeans_model.pkl"), "rb") as f:
        kmeans = pickle.load(f)

    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)

    with open(os.path.join(BASE_DIR, "top_similar_products.pkl"), "rb") as f:
        similar_products = pickle.load(f)

    return kmeans, scaler, similar_products


kmeans, scaler, similar_products = load_models()

# Cluster label mapping
cluster_labels = {
    2: "High-Value",
    3: "Regular",
    0: "Occasional",
    1: "At-Risk"
}

# ===============================
# Page Config
# ===============================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="centered"
)

st.title("🛒 Shopper Spectrum")
st.markdown("### Customer Segmentation & Product Recommendation System")

# Sidebar
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Customer Segmentation", "Product Recommendation"]
)

# ===============================
# HOME
# ===============================
if page == "Home":

    st.header("Welcome")

    st.write("""
This application helps you:

✅ Predict customer segment using RFM values.

✅ Recommend similar products.

Use the left sidebar to navigate.
""")

# ===============================
# CUSTOMER SEGMENTATION
# ===============================
elif page == "Customer Segmentation":

    st.header("Customer Segmentation")

    recency = st.number_input(
        "Recency (days)",
        min_value=0,
        value=30
    )

    frequency = st.number_input(
        "Frequency",
        min_value=1,
        value=5
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0,
        value=500.0
    )

    if st.button("Predict Segment"):

        data = np.array([[recency, frequency, monetary]])

        scaled = scaler.transform(data)

        cluster = kmeans.predict(scaled)[0]

        segment = cluster_labels.get(cluster, "Unknown")

        st.success(f"Predicted Segment : {segment}")

        descriptions = {
            "High-Value": "Recent, frequent and high-spending customers.",
            "Regular": "Steady customers with moderate spending.",
            "Occasional": "Purchase only occasionally.",
            "At-Risk": "Customers likely to churn."
        }

        st.info(descriptions.get(segment, ""))

# ===============================
# PRODUCT RECOMMENDATION
# ===============================
elif page == "Product Recommendation":

    st.header("Product Recommendation")

    products = sorted(similar_products.keys())

    product = st.selectbox(
        "Select Product",
        products
    )

    if st.button("Recommend"):

        if product in similar_products:

            st.success(f"Top recommendations for {product}")

            recommendations = similar_products[product][:5]

            for i, (name, score) in enumerate(recommendations, 1):

                st.write(
                    f"{i}. {name} (Similarity Score : {score:.2f})"
                )

        else:

            st.error("Product not found.")
