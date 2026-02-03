import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Auto Data Analytics Tool", layout="wide")
st.title("📊 Auto Data Analytics Tool")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV file uploaded successfully!")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ✅ DEFINE ONCE (IMPORTANT FIX)
    numeric_cols = df.select_dtypes(include=np.number).columns

    # =========================
    # Cleaning Options
    # =========================
    st.sidebar.header("🧹 Data Cleaning")

    if st.sidebar.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()
        st.sidebar.success("Duplicates removed")

    if st.sidebar.checkbox("Handle Missing Values"):
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
        st.sidebar.success("Missing values handled")

    # =========================
    # Download Cleaned Data
    # =========================
    st.sidebar.header("⬇️ Download Data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.sidebar.download_button(
        label="Download Cleaned CSV",
        data=csv,
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # =========================
    # EDA
    # =========================
    st.sidebar.header("📈 EDA")

    if st.sidebar.checkbox("Show EDA"):
        st.subheader("Exploratory Data Analysis")

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # =========================
    # Correlation Heatmap
    # =========================
    if st.sidebar.checkbox("Show Correlation Heatmap"):
        if len(numeric_cols) > 1:
            st.subheader("🔗 Correlation Heatmap")

            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Correlation heatmap requires at least 2 numeric columns.")

    # =========================
    # Statistical Summary
    # =========================
    if st.sidebar.checkbox("Statistical Summary"):
        st.subheader("📊 Statistical Summary")

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.shape[1] == 0:
            st.warning("No numeric columns available for summary.")
        else:
            summary = pd.DataFrame({
                "Mean": numeric_df.mean(),
                "Median": numeric_df.median(),
                "Std Dev": numeric_df.std(),
                "Min": numeric_df.min(),
                "Max": numeric_df.max(),
                "Count": numeric_df.count()
            })
            st.dataframe(summary)

    # =========================
    # Machine Learning
    # =========================
    st.sidebar.header("🤖 Machine Learning")

    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    if st.checkbox("Train ML Model"):
        st.subheader("🤖 Machine Learning")

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical target
        if y.dtype == "object":
            st.info("Target column is categorical → encoding applied")
            y = LabelEncoder().fit_transform(y)

        # One-hot encode features
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Auto detect problem type
        if pd.Series(y).nunique() <= 10:
            # Classification
            st.subheader("📌 Model Type: Classification")

            model_choice = st.selectbox(
                "Choose Classification Model",
                ["Logistic Regression", "Random Forest"]
            )

            model = (
                LogisticRegression(max_iter=1000)
                if model_choice == "Logistic Regression"
                else RandomForestClassifier(n_estimators=100, random_state=42)
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.2f}")
    if "Random Forest" in model_choice:
     st.subheader("🌟 Feature Importance")

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(feat_df.head(10))

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_df.head(10), ax=ax)
    st.pyplot(fig)
else:
            # Regression
            st.subheader("📌 Model Type: Regression")

            model_choice = st.selectbox(
                "Choose Regression Model",
                ["Linear Regression", "Random Forest Regressor"]
            )

            model = (
                LinearRegression()
                if model_choice == "Linear Regression"
                else RandomForestRegressor(n_estimators=100, random_state=42)
            )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            st.success(f"RMSE: {rmse:,.2f}")
if "Random Forest" in model_choice:
    st.subheader("🌟 Feature Importance")

    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(feat_df.head(10))

    fig, ax = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feat_df.head(10), ax=ax)
    st.pyplot(fig)


