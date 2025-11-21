from flask import Flask, render_template, request
import numpy as np
import pickle
import os # <-- Added os import

app = Flask(__name__)

# --- FIX: Use an absolute path for the model file ---
# Get the directory of the current file (app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")

# Load trained model
try:
    # Use the absolute path
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    model = None
    # This print will appear in the PythonAnywhere error log
    print(f"Error loading model from {MODEL_PATH}: {e}")
# ----------------------------------------------------


@app.route("/")
def home():
    return render_template("predict.html")


@app.route("/predict", methods=["POST"])
def predict():
    # CRITICAL CHECK: Ensure the model loaded successfully
    if model is None:
        return render_template("result.html", prediction="Error: Model failed to load. Please check your PythonAnywhere error log.")
    
    try:
        # Get data from form
        gender = float(request.form.get("gender"))
        age = float(request.form.get("age"))
        hypertension = float(request.form.get("hypertension"))
        heart_disease = float(request.form.get("heart_disease"))
        smoking_history = float(request.form.get("smoking_history"))
        bmi = float(request.form.get("bmi"))
        HbA1c_level = float(request.form.get("HbA1c_level"))
        blood_glucose_level = float(request.form.get("blood_glucose_level"))

        # Arrange data (same order as your training dataset)
        input_data = np.array([
            gender,
            age,
            hypertension,
            heart_disease,
            smoking_history,
            bmi,
            HbA1c_level,
            blood_glucose_level
        ]).reshape(1, -1)

        # Predict
        prediction_raw = model.predict(input_data)[0]

        # convert to text for UI
        prediction = "Diabetic" if prediction_raw == 1 else "Not Diabetic"

        return render_template("result.html", prediction=prediction)

    except Exception as e:
        # Added a clearer error message for the user if form data is bad
        return render_template("result.html", prediction=f"Prediction Error: Could not process input data. Check that all fields are filled correctly. Details: {e}")


if __name__ == "__main__":
    app.run(debug=True)