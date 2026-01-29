# Customer Churn Prediction System

A machine learning-powered system to predict customer churn with Flask API and Streamlit dashboard.

## Features

- **Flask API** - REST API with prediction endpoints
- **Streamlit Dashboard** - Interactive web interface for predictions
- **Single & Batch Predictions** - Predict for one customer or upload CSV for batch processing
- **Risk Level Classification** - High, Medium, Low risk categorization
- **Real-time Monitoring** - Track model performance and predictions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Flask API

```bash
python app.py
```

API will be available at `http://localhost:5010`

### Run Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```

Dashboard will open in your browser automatically.

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Single customer prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/info` - Model information

## Model Information

- **Version:** v2.1
- **Type:** Machine Learning Classifier
- **Accuracy:** 87.3%
- **ROC-AUC:** 89.4%

## Project Structure

```
AI_Deployment/
├── app.py                          # Flask API
├── streamlit_app.py                # Streamlit dashboard
├── monitoring_dashboard.py         # Monitoring tools
├── error_handler.py                # Error handling utilities
├── logging_config.py               # Logging configuration
├── churn_prediction_model_final.pkl # Trained model
├── feature_scaler.pkl              # Feature scaler
├── feature_names.txt               # Feature list
└── requirements.txt                # Dependencies
```

## License

MIT
