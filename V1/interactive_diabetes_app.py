import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans, DBSCAN

st.set_page_config(layout="wide", page_title="Diabetes Data Platform")

# === Helper Functions ===

def save_client_data(data: dict):
    df = pd.DataFrame([data])
    now = datetime.now()
    filename = f"data/client_data_{now.strftime('%Y_%m')}.csv"
    os.makedirs("data", exist_ok=True)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def load_monthly_data():
    os.makedirs("data", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith(".csv")]
    selected = st.sidebar.selectbox("📁 Choose a file", files)
    if selected:
        return pd.read_csv(os.path.join("data", selected))
    return None

def load_external_data(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]
        if file_extension == "csv":
            return pd.read_csv(uploaded_file)
        elif file_extension in ["xls", "xlsx"]:
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a .csv, .xls, or .xlsx file.")
    return None

# === Sidebar Navigation ===
st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio("Choose Mode", ["Client Side", "Admin Side"])

# ====================================================================================
# 🧾 CLIENT SIDE
# ====================================================================================
if app_mode == "Client Side":
    st.title("🧾 Client Data Entry Form")
    st.markdown("Submit basic diabetes parameters. Your data will be saved for analysis.")

    with st.form("client_form", clear_on_submit=True):
        st.subheader("🔢 Core Parameters")
        data = {
            "Pregnancies": st.number_input("Pregnancies", min_value=0, step=1),
            "Glucose": st.number_input("Glucose", min_value=0.0),
            "BloodPressure": st.number_input("BloodPressure", min_value=0.0),
            "SkinThickness": st.number_input("SkinThickness", min_value=0.0),
            "Insulin": st.number_input("Insulin", min_value=0.0),
            "BMI": st.number_input("BMI", min_value=0.0),
            "DiabetesPedigreeFunction": st.number_input("DiabetesPedigreeFunction", min_value=0.0),
            "Age": st.number_input("Age", min_value=0),
            "Outcome": st.selectbox("Outcome (0: No, 1: Yes)", [0, 1])
        }

        st.subheader("🧩 Optional Fields")
        data["PatientID"] = st.text_input("Patient ID (optional)")
        data["Gender"] = st.selectbox("Gender (optional)", ["", "Male", "Female", "Other"])
        data["Timestamp"] = datetime.now().isoformat()

        st.subheader("➕ Add Custom Fields")
        extra_fields = {}
        num_extra = st.number_input("Number of custom fields", 0, 10, 0)
        for i in range(int(num_extra)):
            col1, col2 = st.columns([1, 2])
            with col1:
                key = st.text_input(f"Field Name {i+1}", key=f"key_{i}")
            with col2:
                val = st.text_input(f"Value {i+1}", key=f"val_{i}")
            if key:
                extra_fields[key] = val

        data.update(extra_fields)
        if st.form_submit_button("✅ Submit"):
            save_client_data(data)
            st.success("Data submitted and saved!")

# ====================================================================================
# 📊 ADMIN SIDE
# ====================================================================================
else:
    st.title("📊 Admin Dashboard – Diabetes Data Analysis")

    # File Upload Section
    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel File", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
        df = load_external_data(uploaded_file)
        if df is not None:
            st.success("Data loaded successfully!")
    else:
        df = load_monthly_data()

    if df is not None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        st.subheader("📁 Dataset Preview")
        st.dataframe(df, use_container_width=True)

        st.subheader("🔧 Data Filtering")
        with st.expander("Dynamic Filtering Controls"):
            for col in numeric_cols:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                df = df[df[col].between(*st.slider(f"{col} Range", min_val, max_val, (min_val, max_val)))]


        st.subheader("📈 Visualizations")
        tabs = st.tabs(["Correlation Heatmap", "Scatter Plot", "Feature Importance", "Clustering"])

        with tabs[0]:
            st.write("Correlation between numeric features")
            fig, ax = plt.subplots()
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        with tabs[1]:
            st.write("Scatter Plot")
            x_col = st.selectbox("X Axis", numeric_cols)
            y_col = st.selectbox("Y Axis", numeric_cols, index=1)
            hue_col = st.selectbox("Color By", df.columns)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
            st.pyplot(fig)

        with tabs[2]:
            st.write("Random Forest Feature Importance")
            target = st.selectbox("Target Variable", numeric_cols)
            features = st.multiselect("Features", [col for col in numeric_cols if col != target], default=numeric_cols)
            if st.button("Run Random Forest"):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                importances = model.feature_importances_
                st.bar_chart(pd.Series(importances, index=features))
                accuracy = accuracy_score(y_test, model.predict(X_test))
                st.info(f"Model Accuracy: **{accuracy:.2%}**")

        with tabs[3]:
            st.write("K-Means or DBSCAN Clustering")
            clust_cols = st.multiselect("Clustering Columns", numeric_cols, default=numeric_cols[:2])
            method = st.radio("Clustering Algorithm", ["K-Means", "DBSCAN"])
            scale = StandardScaler().fit_transform(df[clust_cols])
            if method == "K-Means":
                k = st.slider("Number of Clusters", 2, 10, 3)
                model = KMeans(n_clusters=k)
                df["Cluster"] = model.fit_predict(scale)
            else:
                eps = st.slider("Epsilon", 0.1, 5.0, 1.0)
                min_samples = st.slider("Min Samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                df["Cluster"] = model.fit_predict(scale)
            fig, ax = plt.subplots()
            sns.scatterplot(x=clust_cols[0], y=clust_cols[1], hue="Cluster", data=df, ax=ax, palette="tab10")
            st.pyplot(fig)
    else:
        st.warning("Upload or select a data file to begin.")
