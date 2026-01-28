import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_squared_error

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
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
        st.sidebar.success("Missing values handled")

    # =========================
    # Download Cleaned Data
    # =========================
    st.sidebar.header("⬇️ Download Data")

    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df_to_csv(df)

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

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # =========================
    # Machine Learning
    # =========================
    st.sidebar.header("🤖 Machine Learning")

if st.sidebar.checkbox("Train ML Model"):
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------- Classification --------
    if y.nunique() <= 10:
        st.subheader("Model Type: Classification")

        model_choice = st.sidebar.selectbox(
            "Choose Model",
            ["Logistic Regression", "Random Forest"]
        )

        if model_choice == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        st.success(f"Accuracy: {acc:.2f}")

    # -------- Regression --------
    else:
     st.subheader("Model Type: Regression")

    model_choice = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Random Forest"]
    )

    if model_choice == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)   # 👈 YE LINE ZAROORI HAI
    rmse = np.sqrt(mse)

    st.success(f"RMSE: {rmse:,.2f}")


