
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    filename='api_logs.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)

# Load model and artifacts (do this once at startup)
MODEL_VERSION = "v2.1"

try:
    model = joblib.load('churn_prediction_model_final.pkl')
    scaler = joblib.load('feature_scaler.pkl')

    with open('feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]

    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Prediction function
def predict_churn(customer_data):
    """Predict churn for customer"""
    try:
        df = pd.DataFrame([customer_data])

        # Ensure all features
        for fname in feature_names:
            if fname not in df.columns:
                df[fname] = 0
        df = df[feature_names]

        # Predict
        prediction = int(model.predict(df)[0])
        probability = float(model.predict_proba(df)[0, 1])

        # Risk level
        if probability >= 0.7:
            risk_level = "High"
        elif probability >= 0.4:
            risk_level = "Medium"
        else:
            risk_level = "Low"

        return {
            'prediction': prediction,
            'prediction_label': 'Churn' if prediction == 1 else 'Active',
            'churn_probability': probability,
            'risk_level': risk_level,
            'confidence': max(probability, 1 - probability),
            'model_version': MODEL_VERSION,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return {'error': str(e)}

# API Endpoints
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_version': MODEL_VERSION,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        data = request.json

        # Validate
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Predict
        result = predict_churn(data)

        # Log
        logging.info(f"Prediction made: {result.get('risk_level', 'unknown')}")

        return jsonify(result)

    except Exception as e:
        logging.error(f"API error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.json
        customers = data.get('customers', [])

        if not customers:
            return jsonify({'error': 'No customers provided'}), 400

        # Predict for each
        results = []
        for customer in customers:
            result = predict_churn(customer)
            result['customer_id'] = customer.get('customer_id', 'unknown')
            results.append(result)

        logging.info(f"Batch prediction: {len(results)} customers")

        return jsonify({
            'predictions': results,
            'total_count': len(results),
            'high_risk_count': sum(1 for r in results if r.get('risk_level') == 'High')
        })

    except Exception as e:
        logging.error(f"Batch prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Model information endpoint"""
    return jsonify({
        'model_version': MODEL_VERSION,
        'features_count': len(feature_names) if feature_names else 0,
        'model_type': type(model).__name__ if model else 'Unknown'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5010, debug=False)
