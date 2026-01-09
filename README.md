Customer Churn Prediction – AI Assignment

## Project Overview
This project predicts whether a customer will churn based on historical customer data.
The solution includes data preprocessing, machine learning model training, evaluation,
and deployment using a Flask REST API.

---

## Dataset
- Dataset: Telco Customer Churn
- Source: Kaggle
- Target Variable: `Churn`
  - Yes → 1
  - No → 0

---

## Task 1: Data Preprocessing
- Dropped irrelevant identifier column (`customerID`)
- Handled missing values
- Converted `TotalCharges` to numeric
- Encoded target variable
- Applied:
  - StandardScaler for numerical features
  - OneHotEncoder for categorical features

---

## Task 2: Model Building
Two machine learning models were trained and evaluated:

### Logistic Regression (Final Model)
- Stable and interpretable
- Handles class imbalance using `class_weight="balanced"`
- Selected for deployment

### Decision Tree (Comparison Model)
- Captures non-linear relationships
- Used only for performance comparison

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## Task 3: Model Explanation
**Why Logistic Regression?**
- Simple and interpretable
- Stable during deployment
- Performs well on tabular data

**Feature Impact**
- Tenure, MonthlyCharges, Contract type, and Internet services
  strongly influence churn prediction.

**Future Improvements**
- Hyperparameter tuning
- Try Gradient Boosting models
- Add SHAP for explainability
- Deploy using Docker or cloud services

---

## Task 4: API Creation (Flask)
A REST API was built using Flask.

### Endpoints:
- `GET /` → Health check
- `POST /predict` → Returns churn prediction

### Sample Input (JSON):
```json
{
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.5,
  "TotalCharges": 845.0
}
}
### Sample output:
{
  "prediction": "Churn"
}

### Technologies Used
Python
pandas
numpy 
scikit-learn 
Flask
joblib 
requests

---

## Model Evaluation Results

### Logistic Regression Performance

The Logistic Regression model was selected as the **final model for testing and deployment**
due to its stability, interpretability, and consistent performance.

```text
Accuracy : 0.7381
Precision: 0.5043
Recall   : 0.7834
F1-score : 0.6136

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow, starting from
data preprocessing and model training to deployment using a Flask REST API.
The implemented solution is stable, interpretable, and suitable for real-world
customer churn prediction use cases.

---

**Sunith B**  
M.Tech – Artificial Intelligence and Data Science
