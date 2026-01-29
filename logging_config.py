
import logging
from logging.handlers import RotatingFileHandler
import os
from datetime import datetime

def setup_logging(log_dir='logs'):
    """Setup comprehensive logging"""

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler - General logs
    file_handler = RotatingFileHandler(
        f'{log_dir}/app.log',
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # File handler - Error logs
    error_handler = RotatingFileHandler(
        f'{log_dir}/errors.log',
        maxBytes=10*1024*1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_format)
    logger.addHandler(error_handler)

    # File handler - Predictions
    pred_handler = RotatingFileHandler(
        f'{log_dir}/predictions.log',
        maxBytes=10*1024*1024,
        backupCount=10
    )
    pred_handler.setLevel(logging.INFO)
    pred_format = logging.Formatter(
        '%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    pred_handler.setFormatter(pred_format)

    # Create prediction logger
    pred_logger = logging.getLogger('predictions')
    pred_logger.setLevel(logging.INFO)
    pred_logger.addHandler(pred_handler)

    logging.info("Logging system initialized")

    return logger

def log_prediction(customer_id, prediction, probability, risk_level):
    """Log a prediction"""
    pred_logger = logging.getLogger('predictions')
    pred_logger.info(
        f"PREDICTION | Customer: {customer_id} | "
        f"Result: {'CHURN' if prediction == 1 else 'ACTIVE'} | "
        f"Probability: {probability:.4f} | "
        f"Risk: {risk_level}"
    )

def log_error(error_type, error_message, context=None):
    """Log an error with context"""
    logging.error(
        f"ERROR | Type: {error_type} | "
        f"Message: {error_message} | "
        f"Context: {context or 'None'}"
    )

def log_api_request(endpoint, method, duration, status_code):
    """Log API request"""
    logging.info(
        f"API | Endpoint: {endpoint} | "
        f"Method: {method} | "
        f"Duration: {duration:.3f}s | "
        f"Status: {status_code}"
    )

# Initialize logging
if __name__ == "__main__":
    setup_logging()
    logging.info("Logging test successful")
