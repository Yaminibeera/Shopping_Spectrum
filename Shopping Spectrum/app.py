import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ===============================
# Load saved models
# ===============================
@st.cache_resource
def load_models():
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('top_similar_products.pkl', 'rb') as f:
        similar_products = pickle.load(f)
    return kmeans, scaler, similar_products

kmeans, scaler, similar_products = load_models()

# Cluster label mapping (based on our notebook analysis)
cluster_labels = {
    2: 'High-Value',
    3: 'Regular',
    0: 'Occasional',
    1: 'At-Risk'
}

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Shopper Spectrum", page_icon="🛒", layout="centered")

st.title("🛒 Shopper Spectrum")
st.markdown("Customer Segmentation & Product Recommendation System")

# Sidebar navigation
page = st.sidebar.radio("Navigate", ["Home", "Clustering", "Recommendation"])

# ===============================
# HOME PAGE
# ===============================
if page == "Home":
    st.subheader("Welcome!")
    st.write("Use the sidebar to navigate between:")
    st.write("- **Clustering**: Predict a customer's segment based on RFM values")
    st.write("- **Recommendation**: Get similar product suggestions")

# ===============================
# CLUSTERING MODULE
# ===============================
elif page == "Clustering":
    st.header("🎯 Customer Segmentation")
    st.write("Enter customer purchase behavior to predict their segment.")

    recency = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    frequency = st.number_input("Frequency (number of purchases)", min_value=1, value=5)
    monetary = st.number_input("Monetary (total spend)", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):
        input_data = np.array([[recency, frequency, monetary]])
        input_scaled = scaler.transform(input_data)
        cluster = kmeans.predict(input_scaled)[0]
        segment = cluster_labels.get(cluster, "Unknown")

        st.success(f"This customer belongs to: **{segment}**")

        descriptions = {
            'High-Value': "Recent, frequent, and big spenders. Prioritize for loyalty programs.",
            'Regular': "Steady purchasers but not premium. Good candidates for upselling.",
            'Occasional': "Rare, occasional purchases. Consider re-engagement campaigns.",
            'At-Risk': "Haven't purchased in a long time. Target with win-back offers."
        }
        st.info(descriptions.get(segment, ""))

# ===============================
# RECOMMENDATION MODULE
# ===============================
elif page == "Recommendation":
    st.header("🎁 Product Recommendation")
    st.write("Enter a product name to get 5 similar product suggestions.")

    product_list = list(similar_products.keys())
    product_input = st.selectbox("Select or search a product:", product_list)

    if st.button("Get Recommendations"):
        if product_input in similar_products:
            recommendations = similar_products[product_input]
            st.success(f"Top 5 products similar to: **{product_input}**")
            for i, (prod, score) in enumerate(recommendations[:5], 1):
                st.write(f"{i}. {prod}  (similarity: {score:.2f})")
        else:
            st.error("Product not found.")