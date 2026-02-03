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

        numeric_cols = df.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

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

    st.subheader("🤖 Machine Learning")

    if st.checkbox("Train ML Model"):

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode categorical target safely
        if y.dtype == "object":
            st.info("Target column is categorical → encoding applied")
            y = LabelEncoder().fit_transform(y)

        # One-hot encode features
        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Decide problem type automatically
        if pd.Series(y).nunique() <= 10:
            # -------- Classification --------
            st.subheader("📌 Model Type: Classification")

            model_choice = st.selectbox(
                "Choose Classification Model",
                ["Logistic Regression", "Random Forest"]
            )

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier(
                    n_estimators=100, random_state=42
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            st.success(f"Accuracy: {acc:.2f}")

        else:
            # -------- Regression --------
            st.subheader("📌 Model Type: Regression")

            model_choice = st.selectbox(
                "Choose Regression Model",
                ["Linear Regression", "Random Forest Regressor"]
            )

            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)

            st.success(f"RMSE: {rmse:,.2f}")
