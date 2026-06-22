"""Train KMeans clustering and product similarity models for the Streamlit app."""

from pathlib import Path
import pickle

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spectrum.csv"
N_CLUSTERS = 4


def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df_clean = df.copy()

    df_clean = df_clean.dropna(subset=["CustomerID"])
    df_clean = df_clean[~df_clean["InvoiceNo"].astype(str).str.startswith("C")]
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean = df_clean[df_clean["UnitPrice"] > 0]
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]

    return df_clean


def build_rfm(df_clean: pd.DataFrame) -> pd.DataFrame:
    reference_date = df_clean["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    ).reset_index()
    return rfm


def assign_cluster_labels(rfm: pd.DataFrame) -> dict[int, str]:
    """Map KMeans cluster ids to business segments using mean RFM ranks."""
    profiles = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    remaining = set(profiles.index)

    high_value = min(
        remaining,
        key=lambda c: (
            profiles.loc[c, "Recency"],
            -profiles.loc[c, "Frequency"],
            -profiles.loc[c, "Monetary"],
        ),
    )
    remaining.remove(high_value)

    at_risk = max(remaining, key=lambda c: profiles.loc[c, "Recency"])
    remaining.remove(at_risk)

    regular = max(
        remaining,
        key=lambda c: (profiles.loc[c, "Frequency"], profiles.loc[c, "Monetary"]),
    )
    remaining.remove(regular)

    occasional = remaining.pop()

    return {
        int(high_value): "High-Value",
        int(regular): "Regular",
        int(occasional): "Occasional",
        int(at_risk): "At-Risk",
    }


def build_similarity_dict(df_clean: pd.DataFrame) -> dict:
    customer_product_matrix = df_clean.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0,
    )
    product_customer_matrix = customer_product_matrix.T
    product_similarity = cosine_similarity(product_customer_matrix)
    product_similarity_df = pd.DataFrame(
        product_similarity,
        index=product_customer_matrix.index,
        columns=product_customer_matrix.index,
    )

    top_similar_dict = {}
    for product in product_similarity_df.columns:
        similar_scores = product_similarity_df[product].sort_values(ascending=False)
        top_10 = similar_scores.iloc[1:11]
        top_similar_dict[product] = list(zip(top_10.index, top_10.values))

    return top_similar_dict


def main() -> None:
    print("Loading and cleaning data...")
    df_clean = load_and_clean_data()
    print(f"Cleaned rows: {len(df_clean):,}")

    print("Building RFM features...")
    rfm = build_rfm(df_clean)

    print("Training KMeans model...")
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)
    cluster_labels = assign_cluster_labels(rfm)

    print("Building product similarity index...")
    top_similar_dict = build_similarity_dict(df_clean)

    print("Saving artifacts...")
    with open(BASE_DIR / "kmeans_model.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(BASE_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(BASE_DIR / "cluster_labels.pkl", "wb") as f:
        pickle.dump(cluster_labels, f)
    with open(BASE_DIR / "top_similar_products.pkl", "wb") as f:
        pickle.dump(top_similar_dict, f)

    print("Cluster labels:", cluster_labels)
    print(f"Products indexed: {len(top_similar_dict):,}")
    print("Done. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
