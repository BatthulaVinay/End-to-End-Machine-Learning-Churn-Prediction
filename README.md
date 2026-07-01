---
title: Customer Churn Prediction
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.58.0
app_file: app.py
pinned: false
---
# 📊 Customer Churn Prediction 
[View Live App](https://huggingface.co/spaces/BatthulaVinay/customer-churn-prediction)
<!-- Badges Section -->
<p align="left">
  <a href="https://huggingface.co/spaces/BatthulaVinay/customer-churn-prediction">
    <img src="https://img.shields.io/badge/Live_Demo-Link-brightgreen?style=for-the-badge&logo=appveyor" alt="HuggingFace Space">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://www.docker.com/">
    <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" alt="Docker">
  </a>
  <a href="https://fastapi.tiangolo.com/">
    <img src="https://img.shields.io/badge/FastAPI-05998B?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  </a>
</p>
An end-to-end machine learning system designed to predict customer churn, enabling businesses to take proactive retention measures. This project utilizes classification models to analyze customer behavior patterns and provides actionable insights.

# 🔴 Live Demo
[Try it live on HuggingFace Spaces](https://huggingface.co/spaces/BatthulaVinay/customer-churn-prediction)

# ✨ Features

Data Analysis: Visualized churn drivers using correlation heatmaps.
Model Explainability: Used SHAP for transparent prediction insights.
FastAPI Backend: Served low-latency, real-time model predictions.
Docker Ready: Fully containerized for consistent deployment.
Live Demo: Hosted and accessible via Hugging Face Spaces.

### 📊 Model Performance Comparison

| Model | Precision | Recall | ROC-AUC | F1-Score | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** | **0.752** | 0.784 | **0.901** | **0.768** | **0.824** |
| Random Forest | 0.617 | 0.814 | 0.892 | 0.702 | 0.774 |
| Decision Tree | 0.611 | 0.680 | 0.767 | 0.644 | 0.566 |
| LR + SMOTE | 0.250 | 0.907 | 0.811 | 0.392 | 0.421 |
| Standard LR | 0.237 | **0.928** | 0.816 | 0.378 | 0.421 |



