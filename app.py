# Filename: app.py

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the pre-trained model and scaler
try:
    model = joblib.load('cancer_prediction_model.joblib')
    scaler = joblib.load('scaler.joblib')
    # The feature names are crucial for the order of input
    feature_names = [
        'mean radius', 'mean texture', 'mean perimeter', 'mean area',
        'mean smoothness', 'mean compactness', 'mean concavity',
        'mean concave points', 'mean symmetry', 'mean fractal dimension',
        'radius error', 'texture error', 'perimeter error', 'area error',
        'smoothness error', 'compactness error', 'concavity error',
        'concave points error', 'symmetry error', 'fractal dimension error',
        'worst radius', 'worst texture', 'worst perimeter', 'worst area',
        'worst smoothness', 'worst compactness', 'worst concavity',
        'worst concave points', 'worst symmetry', 'worst fractal dimension'
    ]
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None
    feature_names = []  # Initialize empty if loading failed

@app.route('/')
def home():
    return render_template('index.html')

# Define the prediction API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'ML model not loaded. Please check server logs.'}), 500

    try:
        data = request.get_json(force=True)

        # --- SIMULATING INPUT FOR THE 30 FEATURES FROM SIMPLIFIED INPUTS ---
        # IMPORTANT: This is a placeholder for demonstration.
        # In a real app, you would either:
        # 1. Have an ML model trained directly on these simplified inputs.
        # 2. Use a complex feature engineering pipeline to derive the 30 features.
        # Here, we're using values typical for a BENIGN case as the default.

        # These are values from a known BENIGN example in the Breast Cancer Wisconsin dataset
        input_values = {
            'mean radius': 12.18,
            'mean texture': 17.84,
            'mean perimeter': 77.79,
            'mean area': 451.1,
            'mean smoothness': 0.1045,
            'mean compactness': 0.08245,
            'mean concavity': 0.05049,
            'mean concave points': 0.02924,
            'mean symmetry': 0.173,
            'mean fractal dimension': 0.06072,
            'radius error': 0.2823,
            'texture error': 1.107,
            'perimeter error': 2.288,
            'area error': 24.67,
            'smoothness error': 0.005886,
            'compactness error': 0.01561,
            'concavity error': 0.02373,
            'concave points error': 0.007017,
            'symmetry error': 0.01597,
            'fractal dimension error': 0.002853,
            'worst radius': 12.85,
            'worst texture': 23.09,
            'worst perimeter': 82.52,
            'worst area': 514.3,
            'worst smoothness': 0.1412,
            'worst compactness': 0.1652,
            'worst concavity': 0.1893,
            'worst concave points': 0.07334,
            'worst symmetry': 0.2772,
            'worst fractal dimension': 0.07856
        }

        # Get simplified inputs from the frontend
        age = data.get('age', 40)
        family_history = data.get('family_history', 'no')
        lump_noticed = data.get('lump_noticed', 'no')
        unexplained_weight_loss = data.get('unexplained_weight_loss', 'no')
        smoking_history = data.get('smoking_history', 'never')
        alcohol_consumption = data.get('alcohol_consumption', 'none')
        persistent_fatigue = data.get('persistent_fatigue', 'no')
        skin_changes = data.get('skin_changes', 'no')
        chronic_cough = data.get('chronic_cough', 'no')


        # Conditional adjustments based on simplified inputs
        # We'll adjust values towards malignant characteristics IF certain conditions are met
        # The values chosen here are illustrative and still not medically precise mappings.

        # If a lump or unexplained weight loss is noticed (strong indicators for demo)
        if lump_noticed == 'yes' or unexplained_weight_loss == 'yes':
            input_values['mean radius'] = 17.99
            input_values['mean texture'] = 10.38
            input_values['mean perimeter'] = 122.8
            input_values['mean area'] = 1001.0
            input_values['worst radius'] = 25.38
            input_values['worst perimeter'] = 184.6
            input_values['worst area'] = 2019.0
            input_values['worst concave points'] = 0.2654
            input_values['mean concavity'] = 0.3001  # From a malignant example
            input_values['mean compactness'] = 0.2776

        # Family history could slightly increase values
        if family_history == 'yes':
            input_values['mean radius'] = input_values['mean radius'] * 1.05  # 5% increase
            input_values['mean perimeter'] = input_values['mean perimeter'] * 1.05
            input_values['mean area'] = input_values['mean area'] * 1.05

        # Age can be a risk factor, gradually increase size/perimeter
        if age > 50:
            input_values['mean radius'] += (age - 50) * 0.05
            input_values['mean perimeter'] += (age - 50) * 0.05
        if age > 65:
            input_values['mean radius'] += (age - 65) * 0.1
            input_values['mean perimeter'] += (age - 65) * 0.1
            input_values['worst radius'] += (age - 65) * 0.15

        # Smoking history could influence certain aspects related to cell changes (hypothetical)
        if smoking_history == 'current':
            input_values['mean smoothness'] = max(input_values['mean smoothness'], 0.11)
            input_values['mean fractal dimension'] = max(input_values['mean fractal dimension'], 0.07)
        elif smoking_history == 'past':
            input_values['mean smoothness'] = max(input_values['mean smoothness'], 0.105)
            input_values['mean fractal dimension'] = max(input_values['mean fractal dimension'], 0.065)

        # Alcohol consumption (hypothetical mild influence)
        if alcohol_consumption == 'high':
            input_values['mean texture'] = max(input_values['mean texture'], 22.0)

        # Other general symptoms can increase risk (hypothetically)
        if persistent_fatigue == 'yes' or skin_changes == 'yes' or chronic_cough == 'yes':
            input_values['mean texture'] = max(input_values['mean texture'], 20.0)
            input_values['radius error'] = max(input_values['radius error'], 0.4)


        # Create a list in the correct order for the model
        final_input_list = [input_values[feature] for feature in feature_names]

        # Convert input values to a Pandas DataFrame
        input_df = pd.DataFrame([final_input_list], columns=feature_names)

        # Scale the input features using the loaded scaler
        input_scaled = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0]

        # Map prediction to meaningful labels
        result = 'Malignant' if prediction == 1 else 'Benign'

        malignant_proba = prediction_proba[1]  # Probability for class 1 (Malignant)
        benign_proba = prediction_proba[0]    # Probability for class 0 (Benign)

        # Dynamic resolution message based on prediction
        resolution_message = "This prediction is for informational purposes only. Please consult a medical professional for a definitive diagnosis."
        if result == 'Malignant' and malignant_proba > 0.7:  # High confidence malignant
            resolution_message = "Given the high-risk assessment, it is strongly recommended to consult an oncologist or a specialist immediately for further evaluation and diagnostic tests. Do not delay."
        elif result == 'Malignant':  # Lower confidence malignant
            resolution_message = "The assessment indicates a potential risk. We advise you to schedule an appointment with your doctor to discuss these findings and consider further screenings and professional medical advice."
        elif result == 'Benign' and benign_proba < 0.8:  # Lower confidence benign
            resolution_message = "While the assessment suggests a lower risk, consistent monitoring and regular check-ups with your doctor are recommended, especially if you have any concerns or new symptoms. A second opinion may also be considered."
        else:  # High confidence benign
            resolution_message = "The assessment indicates a lower risk. Continue with your regular health screenings and maintain a healthy lifestyle. If new or concerning symptoms arise, consult your doctor promptly."

        response = {
            'prediction': result,
            'probability': f"{malignant_proba * 100:.2f}%" if prediction == 1 else f"{benign_proba * 100:.2f}%",
            'benign_probability': f"{benign_proba * 100:.2f}%",
            'malignant_probability': f"{malignant_proba * 100:.2f}%",
            'message': resolution_message
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Ensure this line is at the far left (no indentation)
