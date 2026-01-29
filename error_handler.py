
class APIError(Exception):
    """Base API error"""
    pass

class ModelLoadError(APIError):
    """Error loading model"""
    pass

class PredictionError(APIError):
    """Error during prediction"""
    pass

class ValidationError(APIError):
    """Input validation error"""
    pass

def handle_error(error, context=None):
    """
    Centralized error handling

    Args:
        error: Exception object
        context: Additional context dict

    Returns:
        dict with error response
    """
    error_response = {
        'error': True,
        'error_type': type(error).__name__,
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }

    if context:
        error_response['context'] = context

    # Log error
    log_error(
        error_type=error_response['error_type'],
        error_message=error_response['message'],
        context=context
    )

    # Determine status code
    if isinstance(error, ValidationError):
        status_code = 400
    elif isinstance(error, ModelLoadError):
        status_code = 503
    elif isinstance(error, PredictionError):
        status_code = 500
    else:
        status_code = 500

    return error_response, status_code

def validate_prediction_input(data):
    """
    Validate prediction input

    Args:
        data: dict with customer data

    Raises:
        ValidationError: if validation fails
    """
    required_fields = ['num_orders', 'total_spent', 'days_since_last_order']

    # Check required fields
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")

    # Validate types and ranges
    numeric_fields = {
        'num_orders': (0, None),
        'total_spent': (0, None),
        'days_since_last_order': (0, None)
    }

    for field, (min_val, max_val) in numeric_fields.items():
        if field in data:
            try:
                value = float(data[field])

                if min_val is not None and value < min_val:
                    raise ValidationError(
                        f"{field} must be >= {min_val}, got {value}"
                    )

                if max_val is not None and value > max_val:
                    raise ValidationError(
                        f"{field} must be <= {max_val}, got {value}"
                    )

            except (ValueError, TypeError):
                raise ValidationError(f"{field} must be numeric")

    return True
