import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

st.set_page_config(page_title="Auto Data Analytics Tool", layout="wide")
st.title("📊 Auto Data Analytics Tool")

# =========================
# Upload CSV
# =========================
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("CSV uploaded successfully")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # =========================
    # Data Cleaning
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
        "Download Cleaned CSV",
        csv,
        "cleaned_data.csv",
        "text/csv"
    )

    # =========================
    # EDA SECTION (TABS)
    # =========================
    st.sidebar.header("📈 EDA")

    if st.sidebar.checkbox("Show EDA"):
        st.subheader("📊 Exploratory Data Analysis")

        numeric_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include=["object", "category"]).columns

        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
            ["📈 Distributions", "🔗 Correlation", "📊 Statistics",
             "📦 Outliers", "🧩 Categorical", "🧠 Insights"]
        )

        # -------- TAB 1: Distributions --------
        with tab1:
            if len(numeric_cols) == 0:
                st.warning("No numeric columns found.")
            else:
                for col in numeric_cols:
                    fig, ax = plt.subplots()
                    sns.histplot(df[col], kde=True, ax=ax)
                    ax.set_title(f"Distribution of {col}")
                    st.pyplot(fig)

        # -------- TAB 2: Correlation --------
        with tab2:
            if len(numeric_cols) < 2:
                st.info("Need at least 2 numeric columns.")
            else:
                corr = df[numeric_cols].corr()
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
                st.pyplot(fig)

        # -------- TAB 3: Statistics --------
        with tab3:
            if len(numeric_cols) == 0:
                st.warning("No numeric columns.")
            else:
                summary = pd.DataFrame({
                    "Mean": df[numeric_cols].mean(),
                    "Median": df[numeric_cols].median(),
                    "Std Dev": df[numeric_cols].std(),
                    "Min": df[numeric_cols].min(),
                    "Max": df[numeric_cols].max()
                })
                st.dataframe(summary)

        # -------- TAB 4: Outliers --------
        with tab4:
            if len(numeric_cols) == 0:
                st.warning("No numeric columns.")
            else:
                for col in numeric_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - 1.5 * IQR
                    upper = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower) | (df[col] > upper)]

                    st.write(f"**{col}** → Outliers: {outliers.shape[0]}")

                    fig, ax = plt.subplots()
                    sns.boxplot(x=df[col], ax=ax)
                    st.pyplot(fig)

        # -------- TAB 5: Categorical --------
        with tab5:
            if len(cat_cols) == 0:
                st.warning("No categorical columns.")
            else:
                col = st.selectbox("Select column", cat_cols)
                counts = df[col].value_counts()

                st.info(
                    f"Most frequent value: **{counts.idxmax()}** "
                    f"({counts.max()} rows)"
                )

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.countplot(y=df[col], order=counts.index, ax=ax)
                st.pyplot(fig)

        # -------- TAB 6: Insights --------
        with tab6:
            rows, cols = df.shape
            st.write(f"Dataset has **{rows} rows** and **{cols} columns**")

            missing = df.isna().sum().sum()
            if missing == 0:
                st.success("No missing values")
            else:
                st.warning(f"{missing} missing values found")

            if len(numeric_cols) > 0:
                high_var = df[numeric_cols].std().idxmax()
                st.info(
                    f"Column **{high_var}** has highest variation "
                    f"(Std = {df[high_var].std():.2f})"
                )

            if len(cat_cols) > 0:
                top_col = cat_cols[0]
                top_val = df[top_col].value_counts().idxmax()
                st.info(
                    f"In **{top_col}**, most common value is **{top_val}**"
                )

    # =========================
    # MACHINE LEARNING
    # =========================
    st.sidebar.header("🤖 Machine Learning")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    if st.checkbox("Train ML Model"):
        X = df.drop(columns=[target_col])
        y = df[target_col]

        if y.dtype == "object":
            y = LabelEncoder().fit_transform(y)
            st.info("Target encoded")

        X = pd.get_dummies(X, drop_first=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Classification vs Regression
        if pd.Series(y).nunique() <= 10:
            st.subheader("📌 Classification")

            model_choice = st.selectbox(
                "Choose Model",
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
            st.subheader("📌 Regression")

            model_choice = st.selectbox(
                "Choose Model",
                ["Linear Regression", "Random Forest"]
            )

            if model_choice == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(
                    n_estimators=100, random_state=42
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, preds))
            st.success(f"RMSE: {rmse:,.2f}")

        # Feature Importance
        if "Random Forest" in model_choice:
            st.subheader("⭐ Feature Importance")

            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(feat_df.head(10))

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(
                data=feat_df.head(10),
                x="Importance",
                y="Feature",
                ax=ax
            )
            st.pyplot(fig)
