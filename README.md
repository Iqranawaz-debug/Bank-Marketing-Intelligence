# Bank Marketing Intelligence

An end-to-end machine learning and analytics application that transforms historical bank marketing campaign data into interactive insights, customer subscription predictions, model explanations, and business-oriented decision support.

The project was built as a complete data science workflow rather than only a model-training notebook. It combines exploratory data analysis, business interpretation, leakage-aware feature selection, model comparison, cross-validation, threshold analysis, explainability, single-customer prediction, batch scoring, and an interactive Streamlit interface.

---

## Project Overview

Banks often run marketing campaigns to encourage customers to subscribe to financial products such as term deposits.

However, contacting every customer is expensive and inefficient.

This project explores the question:

> **Can historical campaign data help identify customers who are more likely to subscribe before a planned marketing call?**

The application allows users to:

- explore historical campaign performance
- analyze customer segments
- investigate subscription patterns
- compare machine learning models
- estimate customer subscription likelihood
- score multiple customers using CSV files
- evaluate model performance
- understand which variables influence predictions
- explore decision-threshold trade-offs

---

## Key Features

### Executive Dashboard

A high-level overview of the dataset, customer response rate, selected machine learning model, and major campaign findings.

The dashboard summarizes:

- total customer records
- overall subscription rate
- selected predictive model
- ROC-AUC
- occupation-level response rates
- previous campaign outcomes
- major historical insights

---

### Exploratory Data Analysis

Interactive exploration of historical customer and campaign data.

Users can filter the dataset using features such as:

- occupation
- marital status
- age
- campaign characteristics

The page dynamically updates:

- selected record count
- subscription rate
- campaign-contact statistics
- visualizations
- filtered records

---

### Business Insights

Historical campaign patterns are translated into interpretable business observations.

Examples include analysis of:

- customer occupation
- contact method
- previous campaign outcome
- customer response behavior

The application deliberately distinguishes between:

**association** and **causation**.

Observed patterns are presented as hypotheses that could guide future experiments rather than proof that a particular intervention will cause higher subscription rates.

---

### Customer Prediction

The application provides an interactive interface for scoring an individual customer.

Users can enter information related to:

- age
- occupation
- marital status
- education
- yearly account balance
- credit default
- housing loan
- personal loan
- contact method
- campaign timing
- previous campaign interactions

The trained model then estimates the customer's subscription likelihood.

---

### Batch Prediction

The batch prediction page allows users to upload a CSV containing multiple customers.

The system:

1. validates the uploaded data
2. applies the saved preprocessing pipeline
3. scores customer records
4. produces prediction probabilities
5. allows the scored results to be downloaded

The model is loaded from saved artifacts and is **not retrained during prediction**.

---

## Machine Learning Workflow

The project follows a reproducible machine learning workflow:

Historical Campaign Data
          ↓
Data Understanding
          ↓
Data Cleaning
          ↓
Train / Test Split
          ↓
Preprocessing Pipeline
          ↓
Cross-Validation
          ↓
Model Comparison
          ↓
Model Selection
          ↓
Threshold Analysis
          ↓
Held-Out Evaluation
          ↓
Explainability
          ↓
Interactive Streamlit Application
