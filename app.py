import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    mean_squared_error,
    r2_score
)

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Auto Data Analytics Tool",
    layout="wide"
)

st.title("🚀 Auto Data Analytics Tool")

# =========================
# CACHE
# =========================
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# =========================
# FILE UPLOAD
# =========================
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file is not None:
    df = load_data(file)
    st.success("Data loaded successfully")

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # =========================
    # CLEANING
    # =========================
    st.sidebar.header("🧹 Cleaning")

    if st.sidebar.checkbox("Remove Duplicates"):
        df = df.drop_duplicates()

    if st.sidebar.checkbox("Handle Missing Values"):
        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # =========================
    # DOWNLOAD CLEANED DATA
    # =========================
    st.sidebar.download_button(
        "⬇️ Download Cleaned Data",
        data=df.to_csv(index=False),
        file_name="cleaned_data.csv",
        mime="text/csv"
    )

    # =========================
    # EDA
    # =========================
    st.sidebar.header("📊 EDA")

    if st.sidebar.checkbox("Show EDA"):
        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include=["object"]).columns

        t1, t2, t3, t4, t5, t6 = st.tabs(
            ["Distributions", "Correlation", "Statistics", "Outliers", "Categorical", "Insights"]
        )

        with t1:
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                st.pyplot(fig)

        with t2:
            if len(num_cols) > 1:
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)

        with t3:
            st.dataframe(df[num_cols].describe())

        with t4:
            for col in num_cols:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax)
                st.pyplot(fig)

        with t5:
            if len(cat_cols) > 0:
                col = st.selectbox("Select categorical column", cat_cols)
                fig, ax = plt.subplots()
                sns.countplot(y=df[col], ax=ax)
                st.pyplot(fig)

        with t6:
            st.info(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")
            st.success("EDA completed successfully")

    # =========================
    # MACHINE LEARNING
    # =========================
    st.sidebar.header("🤖 Machine Learning")

    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

    if st.checkbox("Train Model"):
        X = df.drop(columns=[target])
        y = df[target]

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =========================
        # CLASSIFICATION
        # =========================
        if pd.Series(y).nunique() <= 10:
            st.subheader("📌 Classification")

            model_choice = st.selectbox(
                "Select Model",
                ["Logistic Regression", "Random Forest"]
            )

            if model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = RandomForestClassifier(
                    n_estimators=200,
                    random_state=42
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.metric("Accuracy", accuracy_score(y_test, preds))
            st.metric("Precision", precision_score(y_test, preds, average="weighted"))
            st.metric("Recall", recall_score(y_test, preds, average="weighted"))

        # =========================
        # REGRESSION
        # =========================
        else:
            st.subheader("📌 Regression")

            model_choice = st.selectbox(
                "Select Model",
                ["Linear Regression", "Random Forest"]
            )

            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(
                    n_estimators=200,
                    random_state=42
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            st.metric("RMSE", np.sqrt(mean_squared_error(y_test, preds)))
            st.metric("R² Score", r2_score(y_test, preds))

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.subheader("⭐ Feature Importance")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            importance = np.abs(model.coef_[0])

        fi = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importance
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=fi.head(10),
            x="Importance",
            y="Feature",
            ax=ax
        )
        st.pyplot(fig)

        # =========================
        # SAVE MODEL
        # =========================
        st.subheader("💾 Save Trained Model")

        if st.button("Download Model"):
            with open("model.pkl", "wb") as f:
                pickle.dump(model, f)
            st.success("model.pkl saved successfully")

        # =========================
        # PREDICTION PLAYGROUND
        # =========================
        st.subheader("🎯 Prediction Playground")

        user_input = {}
        for col in X.columns:
            user_input[col] = st.number_input(col, value=0.0)

        if st.button("Predict"):
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)
            st.success(f"Prediction: {prediction[0]}")
