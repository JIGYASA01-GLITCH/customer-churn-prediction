# 📊 Customer Churn Prediction Using Machine Learning

> Predict · Explain · Prioritize · Retain

An end-to-end Machine Learning system that predicts which telecom customers will churn, explains **why** they are at risk using SHAP, and quantifies the **business value** of early intervention.

🔗 **[Live Dashboard →](https://customer-churn-prediction-foplcxnftu7ecrsm6xrees.streamlit.app/)**

---

## 🎯 Problem Statement

Telecom companies lose thousands of customers every month. Acquiring a new customer costs **5–7x more** than retaining an existing one — yet most companies only react after a customer has already left.

This project solves three problems:
- **WHO** will churn — ML prediction with 0.89 AUC-ROC
- **WHY** they will churn — SHAP explainability per customer
- **WHO to save first** — CLV × Risk priority scoring

---

## 💡 Key Results

| Metric | Value |
|--------|-------|
| Model | XGBoost |
| AUC-ROC | **0.89** |
| High Risk Customers | **261** |
| Revenue at Risk (Top 25) | **$61,749** |
| Net ROI (Top 25 intervention) | **1,382%** |

---

## 🔍 What Makes This Different

Most churn projects stop at accuracy scores. This project goes further:

✅ **SHAP Explainability** — not just *who* churns but *why*

✅ **CLV-Weighted Priority Scoring** — save high-value customers first

✅ **Retention ROI Calculator** — quantify exact business value

✅ **Interactive Dashboard** — usable by non-technical teams

---

## 📂 Project Structure

```
customer-churn-prediction/
│
├── 01_EDA.ipynb                 ← Exploratory Analysis + Feature Engineering
├── 03_modeling.ipynb            ← Model Training + Evaluation
├── 04_explainability.ipynb      ← SHAP Analysis
├── 05_business_metrics.ipynb    ← CLV + Priority Score + ROI
├── dashboard.py                 ← Streamlit Dashboard
├── requirements.txt             ← Dependencies
│
├── churn_model.pkl              ← Saved XGBoost Model
├── customer_scorecard.csv       ← Risk Scores + CLV + Priority
├── shap_values.csv              ← SHAP Values per Customer
├── X_test_with_shap.csv         ← Test Data with Explanations
│
└── WA_Fn-UseC_-Telco-Customer-Churn.csv  ← Raw Dataset
```

---

## 📊 Dashboard Pages

### 1️⃣ Overview
- KPI cards — Total customers, High risk count, Revenue at risk
- Risk tier distribution chart
- Churn probability histogram
- Top 3 SHAP insights

### 2️⃣ Customer Risk Scorecard
- All customers ranked by Priority Score
- Filter by risk tier and contract type
- Color coded — 🔴 HIGH, 🟡 MEDIUM, 🟢 LOW

### 3️⃣ Individual Customer Analysis
- Select any customer by priority rank
- SHAP bar chart — why is this customer at risk?
- Recommended business action

### 4️⃣ Retention ROI Calculator
- Adjust customers to contact, retention rate, incentive cost
- Real-time ROI calculation
- ROI curve across intervention sizes

---

## 🧠 Key Findings from SHAP Analysis

1. **Contract Type** is the #1 churn driver — month-to-month customers churn at 3× the rate of annual contract customers
2. **Tenure** — customers in their first 12 months are highest risk
3. **Charge per Service** — high monthly charges with few services = poor value perception = churn
4. **Electronic Check** payment method is a strong churn signal
5. **No Tech Support + No Online Security** = low switching cost = easy to leave

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas, NumPy | Data manipulation |
| Matplotlib, Seaborn | Visualization |
| Scikit-learn | Preprocessing + Evaluation |
| XGBoost | Best performing classifier |
| Imbalanced-learn | SMOTE for class imbalance |
| SHAP | Model explainability |
| Streamlit | Interactive dashboard |
| Pickle | Model serialization |

---

## 📈 Model Performance

| Model | Accuracy | AUC-ROC | F1-Score |
|-------|----------|---------|----------|
| Logistic Regression | ~80% | ~0.85 | ~0.60 |
| Random Forest | ~82% | ~0.87 | ~0.63 |
| **XGBoost** | **~83%** | **~0.89** | **~0.65** |

> **Why AUC-ROC over Accuracy?**
> Dataset has 73.5% non-churners. A model predicting "no churn" for everyone gets 73.5% accuracy — but is completely useless. AUC-ROC measures how well the model separates churners from non-churners regardless of class imbalance.

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/JIGYASA01-GLITCH/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

---

## 📁 Dataset

**IBM Telco Customer Churn Dataset**
- Source: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)
- 7,043 customers, 21 features
- Target: Churn (Yes/No)
- Churn rate: 26.5%

---

## 👩‍💻 Author

**Jigyasa Chaturvedi**
B.Tech — Artificial Intelligence & Data Science
MITS, Gwalior

[![GitHub](https://img.shields.io/badge/GitHub-JIGYASA01--GLITCH-black?logo=github)](https://github.com/JIGYASA01-GLITCH)

---

## 📌 Semester 4 | Mini Project | 2024
