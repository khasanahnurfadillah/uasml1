import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Clustering Panen Padi", layout="wide")

st.title("Aplikasi Clustering Data Panen Padi")
st.write("Clustering menggunakan K-Means berdasarkan luas panen dan tahun.")

# Upload dataset
uploaded_file = st.file_uploader("Upload file CSV / Excel", type=["csv", "xlsx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Preview Dataset")
    st.dataframe(df.head())

    st.subheader("Seleksi Fitur")
    fitur = st.multiselect(
        "Pilih fitur numerik untuk clustering",
        options=["luas_panen_padi_total", "tahun"],
        default=["luas_panen_padi_total", "tahun"]
    )

    X = df[fitur]

    # Standarisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Parameter K-Means")
    k = st.slider("Jumlah Cluster (k)", 2, 6, 3)
    random_state = st.number_input("Random State", value=42)

    if st.button("Proses Clustering"):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        cluster = kmeans.fit_predict(X_scaled)
        df["cluster"] = cluster

        st.success("Clustering berhasil dilakukan")

        st.subheader("Hasil Data dengan Cluster")
        st.dataframe(df)

        st.subheader("Evaluasi Clustering")
        sil = silhouette_score(X_scaled, cluster)
        db = davies_bouldin_score(X_scaled, cluster)

        st.write(f"Silhouette Score: **{sil:.3f}**")
        st.write(f"Davies-Bouldin Index: **{db:.3f}**")

        st.subheader("Visualisasi Cluster")
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            df[fitur[0]],
            df[fitur[1]],
            c=cluster
        )
        ax.set_xlabel(fitur[0])
        ax.set_ylabel(fitur[1])
        ax.set_title("Visualisasi K-Means Clustering")
        st.pyplot(fig)
else:
    st.info("Silakan upload dataset untuk memulai")