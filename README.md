# 🚀 Auto Data Analytics Tool

An end-to-end **Streamlit-based web application** that allows users to upload any CSV dataset and automatically perform **Data Cleaning, Exploratory Data Analysis (EDA), and Machine Learning** — all without writing code.

This project is designed for **students, beginners, analysts, and non-technical users** who want quick insights from data.

---

## ✨ Features

### 📂 Data Handling

* Upload CSV files
* Preview dataset instantly
* Remove duplicates
* Handle missing values automatically
* Download cleaned dataset

### 📊 Exploratory Data Analysis (EDA)

Interactive EDA using **tabs**:

* 📈 Distributions (histograms)
* 🔗 Correlation Heatmap
* 📊 Statistical Summary
* 📦 Outlier Detection (IQR + boxplots)
* 🧩 Categorical Analysis
* 🧠 Auto Insights (rows, columns, missing values, variance)

### 🤖 Machine Learning (Auto-detect)

* Automatic **Classification / Regression detection**
* Models supported:

  * Logistic Regression
  * Linear Regression
  * Random Forest (Classifier & Regressor)

### 📐 Model Evaluation

* **Classification**:

  * Accuracy
  * Precision
  * Recall
* **Regression**:

  * RMSE
  * R² Score

### ⭐ Feature Importance

* Random Forest: feature_importances_
* Logistic / Linear: coefficient-based importance

### 💾 Advanced (Phase 4)

* Save trained model (`model.pkl`)
* Prediction Playground (manual input → prediction)
* Streamlit caching for performance

---

## 🛠 Tech Stack

* **Frontend / App**: Streamlit
* **Data Processing**: Pandas, NumPy
* **Visualization**: Matplotlib, Seaborn
* **Machine Learning**: Scikit-learn
* **Language**: Python 3

---

## ▶️ How to Run Locally

### 1️⃣ Clone the repository

```bash
git clone https://github.com/krishna2006pathak/auto-data-analytics-tool.git
cd auto-data-analytics-tool
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the app

```bash
streamlit run app.py
```

The app will open automatically in your browser 🌐

---

## 🌍 Deployment

The app can be deployed easily on **Streamlit Cloud**.
Just connect the GitHub repository and select `app.py` as the main file.

---

## 🎯 Use Cases

* Students learning Data Analytics / ML
* Quick analysis of CSV datasets
* Interview-ready data project
* No-code ML experimentation

---

## 👤 Author

**Krishna Pathak**
Aspiring Data Analyst / ML Engineer
GitHub: [https://github.com/krishna2006pathak](https://github.com/krishna2006pathak)

---

## ⭐ If you like this project

Give it a ⭐ on GitHub — it motivates continuous improvement!
