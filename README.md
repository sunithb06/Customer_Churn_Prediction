# Customer Churn Prediction – AI Assignment

**Author:** Sunith B
**Program:** M.Tech – Artificial Intelligence and Data Science

---

## 1. Project Overview

This project focuses on predicting whether a customer is likely to churn (leave the service) based on historical customer data. The solution demonstrates an end-to-end machine learning pipeline, including data preprocessing, model training, evaluation, and deployment through a Flask-based REST API.

The goal is to build a **stable, interpretable, and deployable** churn prediction system suitable for real-world business applications.

---

## 2. Dataset

* **Dataset Name:** Telco Customer Churn
* **Source:** Kaggle
* **Target Variable:** `Churn`

  * `Yes` → 1
  * `No` → 0

The dataset contains customer demographic information, account details, and service usage patterns.

---

## 3. Task 1: Data Preprocessing

The following preprocessing steps were applied to prepare the data for modeling:

* Dropped irrelevant identifier column: `customerID`
* Handled missing values appropriately
* Converted `TotalCharges` from string to numeric format
* Encoded the target variable (`Churn`) into binary values

### Feature Transformation

* **Numerical Features:** Scaled using `StandardScaler`
* **Categorical Features:** Encoded using `OneHotEncoder`

These steps ensured that all features were in a suitable format for machine learning algorithms.

---

## 4. Task 2: Model Building

Two machine learning models were trained and evaluated:

### 4.1 Logistic Regression (Final Model)

* Simple and highly interpretable
* Stable during deployment
* Handles class imbalance using `class_weight="balanced"`
* Selected as the **final model** for deployment

### 4.2 Decision Tree (Comparison Model)

* Captures non-linear relationships
* Used only for performance comparison
* Not selected for deployment due to lower stability

---

## 5. Model Evaluation

The models were evaluated using the following metrics:

* Accuracy
* Precision
* Recall
* F1-score

### Logistic Regression Performance

The Logistic Regression model showed consistent and reliable performance and was selected as the final model.

```
Accuracy : 0.7381
Precision: 0.5043
Recall   : 0.7834
F1-score : 0.6136
```

---

## 6. Task 3: Model Explanation

### Why Logistic Regression?

* Easy to interpret and explain to stakeholders
* Less prone to overfitting on tabular data
* Performs well for binary classification problems
* Suitable for real-time API-based deployment

### Feature Impact

Key features influencing churn prediction include:

* Customer tenure
* Monthly charges
* Contract type
* Internet service type

These features strongly affect a customer’s likelihood of churning.

### Future Improvements

* Hyperparameter tuning
* Experiment with Gradient Boosting models (XGBoost, LightGBM)
* Add SHAP for better explainability
* Deploy using Docker or cloud platforms (AWS, GCP, Azure)

---

## 7. Task 4: API Creation (Flask)

A RESTful API was developed using Flask to serve the trained model.

### API Endpoints

* `GET /` → Health check
* `POST /predict` → Returns churn prediction

### Sample Input (JSON)

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
```

### Sample Output

```json
{
  "prediction": "Churn"
}
```

---

## 8. Technologies Used

* Python
* pandas
* numpy
* scikit-learn
* Flask
* joblib
* requests

---

## 9. Conclusion

This project successfully demonstrates a complete machine learning workflow, from data preprocessing and model training to evaluation and deployment using a Flask REST API. The final solution is interpretable, stable, and suitable for real-world customer churn prediction scenarios.

All assignment tasks, namely **Task 1, Task 2, Task 3, Task 4, and Task 5**, have been successfully completed as per the given requirements.

