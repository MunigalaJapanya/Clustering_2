# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clustering Pipeline", layout="wide")
st.title("🌍 World Development Data - Clustering Dashboard")

uploaded_file = st.sidebar.file_uploader("📂 Upload Excel File", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    st.info("Using default dataset.")
    df = pd.read_excel("World_development_mesurement.xlsx")

st.write("### Data Preview", df.head())

# Select numeric columns
X = df.select_dtypes(include=[np.number]).copy()
X = X.fillna(X.median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.success(f"Using {X.shape[1]} numeric features for clustering.")

# Sidebar parameters
method = st.sidebar.selectbox("Choose Clustering Algorithm", ["KMeans", "Agglomerative", "DBSCAN"])

if method == "KMeans":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
    model = KMeans(n_clusters=k, random_state=42)
    labels = model.fit_predict(X_scaled)
elif method == "Agglomerative":
    k = st.sidebar.slider("Number of Clusters (k)", 2, 10, 4)
    linkage_method = st.sidebar.selectbox("Linkage", ["ward", "complete", "average", "single"])
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage_method)
    labels = model.fit_predict(X_scaled)
else:
    eps = st.sidebar.slider("Epsilon (eps)", 0.5, 3.0, 1.0, 0.1)
    min_samples = st.sidebar.slider("Min Samples", 3, 10, 5, 1)
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X_scaled)

df["Cluster"] = labels

# Compute metrics
if len(set(labels)) > 1 and (np.sum(labels != -1) > 1):
    sil = silhouette_score(X_scaled, labels)
    ch = calinski_harabasz_score(X_scaled, labels)
    db = davies_bouldin_score(X_scaled, labels)
else:
    sil = ch = db = np.nan

st.write("### 🧮 Cluster Evaluation Metrics")
st.write(f"**Silhouette Score:** {sil:.4f}")
st.write(f"**Calinski-Harabasz Score:** {ch:.2f}")
st.write(f"**Davies-Bouldin Score:** {db:.2f}")

# Cluster Summary
st.write("### 📊 Cluster Summary")
st.dataframe(df["Cluster"].value_counts().rename_axis("Cluster").reset_index(name="Count"))

# Visualization (PCA projection)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_plot = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_plot["Cluster"] = labels

plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Cluster", palette="tab10", s=40)
plt.title(f"{method} Clustering (2D PCA Projection)")
st.pyplot(plt)

st.download_button("📥 Download Clustered Data", df.to_csv(index=False).encode('utf-8'), "clustered_data.csv", "text/csv")
