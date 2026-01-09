from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "churn_logistic_pipeline.pkl")
model = joblib.load(MODEL_PATH)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Customer Churn Prediction API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])

    prediction = model.predict(input_df)[0]
    result = "Churn" if prediction == 1 else "No Churn"

    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)
