# Healthcare-EDA
An end-to-end machine learning project that performs Exploratory Data Analysis (EDA) on healthcare records and uses a Random Forest model deployed via Flask to predict patient test results.
# 🏥 Healthcare Test Results Predictor: EDA & Machine Learning Web App

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-lightgrey.svg)
![Scikit-Learn](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange.svg)

## 📌 Project Overview
This is a full-stack data science and machine learning project that predicts a patient's medical test results (Normal, Inconclusive, or Abnormal) based on their clinical and demographic data. 

The project is divided into two main phases:
1. **Exploratory Data Analysis (EDA) & Model Training:** Conducted in Google Colab to clean data, visualize healthcare trends, and train a Random Forest Classifier.
2. **Web Application Deployment:** A local web server built with Flask and PyCharm, providing a user-friendly HTML/CSS interface for medical professionals to input patient data and receive real-time predictions.

## 📊 Dataset
The model was trained on `healthcare.csv`, utilizing the following key features:
* **Age:** Numerical (e.g., 45, 60)
* **Gender:** Male, Female
* **Blood Type:** A+, A-, B+, B-, AB+, AB-, O+, O-
* **Medical Condition:** Cancer, Obesity, Diabetes, Asthma, Hypertension, Arthritis
* **Admission Type:** Urgent, Emergency, Elective
* **Medication:** Paracetamol, Ibuprofen, Aspirin, Penicillin, Lipitor

**Target Variable:** `Test Results` (Multi-class: 0 = Normal, 1 = Inconclusive, 2 = Abnormal)

## 🛠️ Technologies Used
* **Data Science & EDA:** Python, Pandas, NumPy, Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pickle (Model Serialization)
* **Backend Framework:** Flask (WSGI Web Server)
* **Frontend:** HTML5, CSS3

## 📂 Project Structure
```text
HealthcareApp/
│
├── app.py                  # Main Flask application and routing logic
├── healthcare_model.pkl    # Serialized Random Forest machine learning model
│
├── templates/              
│   └── index.html          # Frontend web interface with form validation
│
└── README.md               # Project documentation
