# SalesIQ: AI-Powered Sales Analytics and Forecasting Platform

SalesIQ is an AI-powered sales analytics platform that helps users analyze sales data, forecast future sales, detect abnormal sales patterns, and generate business analyst reports using Google Gemini AI.

The project is designed as an end-to-end data science and GenAI application, combining data preprocessing, analytics dashboards, machine learning-based forecasting, anomaly detection, and AI-generated business reporting.

---

## Features

- CSV sales dataset upload
- Data cleaning and preprocessing
- Interactive dashboard with filters
- Date range, region, and category filtering
- KPI cards for sales, profit, orders, and average order value
- Executive summary cards
- Monthly sales trend analysis
- Category-wise sales analysis
- Region-wise sales analysis
- Top product analysis
- Sales forecasting
- Forecast evaluation using MAE, RMSE, and MAPE
- Actual vs predicted sales visualization
- Sales anomaly detection
- Business insights generation
- Google Gemini AI-powered business analyst report
- Download filtered data
- Download AI-generated report

---

## Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Google Gemini API

---

## Project Modules

### 1. Data Preprocessing

The preprocessing module cleans the uploaded CSV file by:

- Removing extra spaces from column names
- Converting date columns into proper datetime format
- Converting numeric columns such as Sales, Profit, and Quantity
- Removing invalid rows where required fields are missing

### 2. Dashboard and Analytics

The dashboard provides a visual overview of sales performance using:

- KPI cards
- Sales trend charts
- Category-wise sales charts
- Region-wise sales charts
- Top product analysis
- Data quality checks

### 3. Sales Forecasting

The forecasting module uses historical sales data to predict future sales.

It also provides model evaluation metrics:

- MAE: Mean Absolute Error
- RMSE: Root Mean Squared Error
- MAPE: Mean Absolute Percentage Error

### 4. Anomaly Detection

The anomaly detection module identifies unusual sales spikes or drops using statistical analysis.

This helps businesses detect:

- Sudden sales drops
- Unexpected sales spikes
- Possible seasonal effects
- Data irregularities

### 5. Business Insights

The insights module automatically highlights:

- Best-performing category
- Best-performing region
- Profitability status
- Business improvement suggestions

### 6. Gemini AI Business Report

The AI Report module uses Google Gemini AI to generate a professional natural-language business report from sales metrics.

The report includes:

- Overall sales performance
- Profitability analysis
- Best-performing category, region, and product
- Risks or concerns
- Recommended business actions

---

## Dataset

The project works best with a sales dataset containing columns such as:

- Order Date
- Sales
- Profit
- Quantity
- Category
- Sub-Category
- Region
- Product Name

A Superstore Sales dataset is suitable for this project.

---

## Folder Structure

```text
SalesIQ/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── secrets.toml
│   └── secrets.toml.example
├── data/
│   └── sample_sales.csv
├── utils/
│   ├── preprocessing.py
│   ├── forecasting.py
│   ├── anomaly.py
│   ├── insights.py
│   ├── genai_summary.py
│   └── recommendations.py
├── outputs/
└── presentation/