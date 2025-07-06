# Sales Forecasting using Predictive Analytics

This project implements a **machine learning-based sales forecasting tool** using historical sales data. By applying models like **Linear Regression**, **Decision Trees**, and **Random Forests**, the goal is to accurately **predict future sales** based on key product and transaction features.

## 🔍 Project Overview

**Objective**: Build a predictive model to forecast product sales using machine learning algorithms.

**Key Steps:**
- Data cleaning and preprocessing
- Feature engineering and encoding
- Model training and evaluation (Linear Regression, Decision Tree, Random Forest)
- Performance comparison using RMSE and R² scores
- Visual analysis and feature importance
- Predict sales for new, unseen data

## 📁 Dataset

The dataset used is `sales_data_sample.csv`, which includes historical order and product-level data.

### Features used:
- `QUANTITYORDERED`  
- `PRICEEACH`  
- `ORDERLINENUMBER`  
- `MONTH_ID`  
- `YEAR_ID`  
- `PRODUCTLINE`
- `DEALSIZE`   

### Target:
- `SALES` – The actual sales value to be predicted.

## 📦 Dependencies
Install the required Python packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
