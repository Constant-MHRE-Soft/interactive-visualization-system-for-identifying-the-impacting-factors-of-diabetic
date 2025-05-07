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

# === Page Config ===
st.set_page_config(page_title="Diabetes Platform", layout="wide")


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


def load_latest_file():
    os.makedirs("data", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith(".csv")]
    if not files:
        return None
    latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join("data", x)))
    return pd.read_csv(os.path.join("data", latest_file))


def load_external_data(uploaded_file):
    if uploaded_file is not None:
        ext = uploaded_file.name.split(".")[-1]
        if ext == "csv":
            return pd.read_csv(uploaded_file)
        elif ext in ["xls", "xlsx"]:
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
    return None


def load_all_data():
    os.makedirs("data", exist_ok=True)
    files = [f for f in os.listdir("data") if f.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join("data", f)) for f in files]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


# === Navigation ===
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to", ["📊 Dashboard", "🧾 Client Data Entry", "🧮 Admin Analysis"])

# ====================================================================================
# 📊 DASHBOARD
# ====================================================================================
if page == "📊 Dashboard":
    st.markdown(
        "<h1 style='text-align: center;'>🩺 An interactive visualization system to understand patterns for identifying the impacting factors of diabetic diseases </h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center;'>A simple dashboard showing trends and insights from collected diabetes data</p>",
        unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: right;'>📅 {datetime.now().strftime('%B %d, %Y - %I:%M %p')}</p>",
                unsafe_allow_html=True)

    df = load_all_data()

    if df.empty:
        st.warning("No data available yet. Please add client data first.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        # === Key Stats ===
        col1, col2, col3 = st.columns(3)
        col1.metric("👥 Total Records", len(df))
        col2.metric("🧬 Avg Glucose Level", f"{df['Glucose'].mean():.1f}")
        col3.metric("📏 Avg BMI", f"{df['BMI'].mean():.1f}")

        # === Visuals ===
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔗 Relationship Between Factors")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="Blues", ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("### 📈 Blood Sugar by Age")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x="Age", y="Glucose", hue="Outcome", palette="coolwarm", ax=ax)
            ax.set_title("Glucose vs Age (by Diabetes Result)")
            st.pyplot(fig)

        st.markdown("### ⭐ Key Health Factors")
        target = "Outcome"
        features = [col for col in numeric_cols if col != target]
        if len(features) >= 2:
            X = df[features]
            y = df[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

            fig, ax = plt.subplots(figsize=(10, 4))
            importance.plot(kind='bar', ax=ax)
            ax.set_title("Most Influential Health Indicators")
            st.pyplot(fig)
            acc = accuracy_score(y_test, model.predict(X_test))
            st.info(f"Prediction Accuracy: {acc:.2%}")

        st.markdown("### 🧩 Similar Health Groups")
        if len(features) >= 2:
            scaled = StandardScaler().fit_transform(df[features])
            model = KMeans(n_clusters=3)
            df["Group"] = model.fit_predict(scaled)

            fig, ax = plt.subplots(figsize=(10, 4))
            sns.scatterplot(x=df[features[0]], y=df[features[1]], hue=df["Group"], palette="Set2", ax=ax)
            ax.set_title("Grouped by Health Similarities")
            st.pyplot(fig)

# ====================================================================================
# 🧾 CLIENT SIDE
# ====================================================================================
elif page == "🧾 Client Data Entry":
    st.title("🧾 Client Data Entry Form")
    st.markdown("Submit diabetes-related health data below.")

    with st.form("client_form", clear_on_submit=True):
        st.subheader("🔢 Core Health Info")
        data = {
            "Pregnancies": st.number_input("Pregnancies (by week)", min_value=0, step=1),
            "Glucose": st.number_input("Glucose", min_value=0.0),
            "BloodPressure": st.number_input("BloodPressure", min_value=0.0),
            "SkinThickness": st.number_input("SkinThickness", min_value=0.0),
            "Insulin": st.number_input("Insulin", min_value=0.0),
            "BMI": st.number_input("BMI", min_value=0.0),
            "DiabetesPedigreeFunction": st.number_input("DiabetesPedigreeFunction", min_value=0.0),
            "Age": st.number_input("Age", min_value=0),
            "Outcome": st.selectbox("Diabetes Outcome (0: No, 1: Yes)", [0, 1])
        }

        st.subheader("🧩 Optional Info")
        data["PatientID"] = st.text_input("Patient ID (optional)")
        data["Gender"] = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        data["Timestamp"] = datetime.now().isoformat()

        st.subheader("➕ Add Custom Fields")
        num_extra = st.number_input("Number of custom fields", 0, 10, 0)
        for i in range(int(num_extra)):
            key = st.text_input(f"Field Name {i + 1}", key=f"k_{i}")
            val = st.text_input(f"Value {i + 1}", key=f"v_{i}")
            if key:
                data[key] = val

        if st.form_submit_button("✅ Submit"):
            save_client_data(data)
            st.success("Data submitted and saved!")

# ====================================================================================
# 🧮 ADMIN SIDE
# ====================================================================================
elif page == "🧮 Admin Analysis":
    st.title("📊 Admin – Detailed Data Analysis")

    uploaded_file = st.sidebar.file_uploader("📁 Upload CSV or Excel", type=["csv", "xls", "xlsx"])

    if uploaded_file:
        df = load_external_data(uploaded_file)
    else:
        st.info("No file uploaded — automatically loading latest available data.")
        df = load_latest_file()

    if df is not None and not df.empty:
        st.subheader("📁 Data Preview")
        st.dataframe(df, use_container_width=True)

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.subheader("🔍 Filter Data")
        with st.expander("Filter by range"):
            for col in numeric_cols:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                df = df[df[col].between(*st.slider(f"{col} Range", min_val, max_val, (min_val, max_val)))]

        st.subheader("📈 Visual Exploration")
        tab1, tab2, tab3 = st.tabs(["📉 Glucose vs Age", "🔢 Key Indicators", "🧩 Clustering"])

        with tab1:
            x = "Age"
            y = "Glucose"
            hue = st.selectbox("Color by", df.columns, index=0)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
            st.pyplot(fig)

        with tab2:
            target = st.selectbox("Target", numeric_cols)
            features = st.multiselect("Features", [c for c in numeric_cols if c != target], default=numeric_cols)
            if st.button("Run Random Forest"):
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
                model = RandomForestClassifier()
                model.fit(X_train, y_train)
                importances = pd.Series(model.feature_importances_, index=features)
                st.bar_chart(importances.sort_values(ascending=False))
                acc = accuracy_score(y_test, model.predict(X_test))
                st.info(f"Accuracy: {acc:.2%}")

        with tab3:
            cluster_cols = st.multiselect("Cluster on", numeric_cols, default=numeric_cols[:2])
            algo = st.radio("Algorithm", ["K-Means", "DBSCAN"])
            scaled = StandardScaler().fit_transform(df[cluster_cols])
            if algo == "K-Means":
                k = st.slider("Clusters", 2, 10, 3)
                model = KMeans(n_clusters=k)
                df["Cluster"] = model.fit_predict(scaled)
            else:
                eps = st.slider("Epsilon", 0.1, 5.0, 1.0)
                min_samp = st.slider("Min Samples", 2, 10, 3)
                model = DBSCAN(eps=eps, min_samples=min_samp)
                df["Cluster"] = model.fit_predict(scaled)

            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=cluster_cols[0], y=cluster_cols[1], hue="Cluster", ax=ax, palette="tab10")
            st.pyplot(fig)
    else:
        st.warning("Please upload or select a data file to continue.")
