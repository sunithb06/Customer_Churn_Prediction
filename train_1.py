import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


data = pd.read_csv("data/churn.csv")
print("Dataset loaded:", data.shape)

if "customerID" in data.columns:
    data.drop("customerID", axis=1, inplace=True)


data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")


for col in data.columns:
    if data[col].dtype == "object":
        data[col].fillna(data[col].mode()[0], inplace=True)
    else:
        data[col].fillna(data[col].mean(), inplace=True)


data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

X = data.drop("Churn", axis=1)
y = data["Churn"]

categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ]
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n===== Logistic Regression =====")
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))



os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/churn_logistic_pipeline.pkl")

print("\n✅ Clean Logistic Regression pipeline saved")
