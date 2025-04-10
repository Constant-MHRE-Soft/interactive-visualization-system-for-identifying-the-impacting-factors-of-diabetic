import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN

st.set_page_config(layout="wide")
st.title("🔍 Interactive Visualization System for Diabetes Analysis")

# Sidebar – Upload Dataset
with st.sidebar:
    st.header("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    st.markdown("✅ *Upload diabetes-related data with numeric features and a target column*")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data uploaded successfully!")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    st.subheader("📊 Dataset Preview")
    with st.expander("Expand to view dataset"):
        st.dataframe(df)

    # === Dynamic Filtering Sliders ===
    st.sidebar.header("🔧 Filter Data")
    filter_column = st.sidebar.selectbox("Select column to filter", numeric_cols)
    min_val = float(df[filter_column].min())
    max_val = float(df[filter_column].max())
    selected_range = st.sidebar.slider(f"Select range of {filter_column}", min_val, max_val, (min_val, max_val))
    df = df[(df[filter_column] >= selected_range[0]) & (df[filter_column] <= selected_range[1])]

    # === Correlation Heatmap ===
    if st.checkbox("📈 Show Correlation Heatmap"):
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    # === Scatter Plot ===
    with st.expander("📍 Scatter Plot"):
        st.subheader("Visualize Relationships")
        x_axis = st.selectbox("X-Axis", numeric_cols)
        y_axis = st.selectbox("Y-Axis", numeric_cols, index=1)
        color_by = st.selectbox("Color By (categorical or target column)", df.columns)

        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=df[color_by], palette="Set2", ax=ax)
        st.pyplot(fig)

    # === Feature Importance ===
    with st.expander("📌 Feature Importance (Random Forest)"):
        st.subheader("ML Feature Importance")
        target_col = st.selectbox("Select target column", numeric_cols)
        input_cols = st.multiselect("Select features", numeric_cols, default=numeric_cols)

        if st.button("Run Feature Analysis"):
            X = df[input_cols]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier(n_estimators=100)
            model.fit(X_train, y_train)

            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Feature': input_cols, 'Importance': importances}).sort_values(by="Importance", ascending=False)

            st.write("🔍 **Top Important Features**")
            st.bar_chart(imp_df.set_index("Feature"))

            accuracy = accuracy_score(y_test, model.predict(X_test))
            st.info(f"✅ Model Accuracy: **{accuracy:.2f}**")

    # === Clustering Section ===
    with st.expander("🔗 Clustering (K-Means & DBSCAN)"):
        st.subheader("Unsupervised Clustering")

        clustering_cols = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])
        clustering_data = df[clustering_cols].dropna()
        scale = StandardScaler()
        X_scaled = scale.fit_transform(clustering_data)

        cluster_method = st.radio("Choose clustering algorithm", ["K-Means", "DBSCAN"])

        if cluster_method == "K-Means":
            k = st.slider("Number of clusters", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42)
            df["Cluster"] = kmeans.fit_predict(X_scaled)
        else:
            eps = st.slider("Epsilon (DBSCAN)", 0.1, 5.0, 1.0)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            db = DBSCAN(eps=eps, min_samples=min_samples)
            df["Cluster"] = db.fit_predict(X_scaled)

        st.write("📌 Clustered Results")
        st.dataframe(df[clustering_cols + ["Cluster"]].head())

        fig, ax = plt.subplots()
        sns.scatterplot(x=clustering_cols[0], y=clustering_cols[1], hue="Cluster", data=df, palette="tab10", ax=ax)
        st.pyplot(fig)

else:
    st.warning("Please upload a CSV file to get started.")
