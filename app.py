from pathlib import Path
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
from functools import wraps
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from generate_recommendations import RecommendationGenerator
from path_config import setup_paths

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

class ValidationError(Exception):
    pass

def validate_request_data(data: Dict[str, Any]) -> None:
    """Validate incoming request data."""
    required_fields = {
        'age': int,
        'gender': str,
        'favourite_genres': list,
        'favourite_artists': list,
        'favourite_music': list
    }
    
    for field, field_type in required_fields.items():
        if field not in data:
            raise ValidationError(f"Missing required field: {field}")
        if not isinstance(data[field], field_type):
            raise ValidationError(f"Invalid type for {field}: expected {field_type.__name__}")

def validate_request(f):
    """Request validation decorator."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            if not request.is_json:
                return jsonify({'error': 'Content-Type must be application/json'}), 400
            
            data = request.get_json()
            validate_request_data(data)
            return f(*args, **kwargs)
        except ValidationError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Request validation error: {str(e)}")
            return jsonify({'error': 'Internal server error'}), 500
    return decorated_function

def find_file(paths: List[Path], file_type: str) -> Path:
    """Find the first existing file from a list of possible paths."""
    for path in paths:
        if path.exists():
            logger.info(f"Found {file_type} at: {path}")
            return path
            
    # Log all attempted paths
    logger.error(f"Could not find {file_type}. Attempted paths:")
    for path in paths:
        logger.error(f"  - {path}")
    raise FileNotFoundError(f"{file_type} not found")

# Setup paths
paths = setup_paths()
logger.info(f"Initialized paths: {paths}")

# Define multiple possible paths for each resource
model_paths = [
    paths['models'] / 'best_model.pth',
    paths['models'] / 'best_model_cv.pth',
    paths['models'] / 'latest_checkpoint.pt'
]

data_paths = [
    paths['test_data'],
    paths['processed_data'] / 'test_data.csv'
]

encoder_paths = [
    paths['encoder'],
    paths['processed_data'] / 'encoder.pt'
]

# Initialize model and data
try:
    # Find valid paths for each resource
    model_path = find_file(model_paths, "model file")
    data_path = find_file(data_paths, "data file")
    encoder_path = find_file(encoder_paths, "encoder file")
    
    # Load catalog data
    catalog_data = pd.read_csv(data_path)
    logger.info(f"Loaded catalog with {len(catalog_data)} items")
    
    # Add missing columns for compatibility
    if 'main_genre' not in catalog_data.columns:
        logger.warning("Adding main_genre column")
        catalog_data['main_genre'] = 'Other'
    
    if 'explicit' not in catalog_data.columns:
        logger.warning("Adding explicit column")
        catalog_data['explicit'] = False
    
    # Initialize recommendation generator
    recommender = RecommendationGenerator(
        model_path=str(model_path),
        catalog_data=catalog_data,
        encoders_path=str(encoder_path)
    )
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/routes', methods=['GET'])
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "url": str(rule)
        })
    return {"routes": routes}, 200

@app.route('/')
def home():
    return "Welcome to the app!"

@app.route('/api/recommendations', methods=['POST'])
@validate_request
def get_recommendations():
    try:
        logger.info("Request received")
        request_data = request.get_json()
        logger.info(f"Request data: {request_data}")
        
        # Prepare user data
        user_data = {
            'age': int(request_data['age']),
            'gender': request_data['gender'].upper(),
            'main_genre': request_data['favourite_genres'][0] if request_data['favourite_genres'] else 'Other',
            'music': request_data['favourite_music'],
            'artist_name': request_data['favourite_artists']
        }
        
        # Add required numerical features with defaults
        for feature in recommender.encoders.numerical_features:
            if feature not in user_data and feature != 'age':
                user_data[feature] = 0.0
        
        logger.info("Generating recommendations...")
        recommendations = recommender.generate_recommendations(
            user_data,
            n_recommendations=10
        )
        logger.info("Recommendations generated")
        
        response = {
            'status': 'success',
            'recommendations': recommendations.to_dict(orient='records'),
            'metadata': {
                'user_profile': {
                    'age': user_data['age'],
                    'gender': user_data['gender'],
                    'preferred_genre': user_data['main_genre']
                }
            }
        }
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    try:
        return jsonify({
            'status': 'healthy',
            'model_loaded': True,
            'catalog_size': len(catalog_data),
            'encoders_loaded': hasattr(recommender, 'encoders'),
            'api_version': '1.0.0',
            'supported_genres': list(recommender.encoders.known_genres) if hasattr(recommender, 'encoders') else []
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

if __name__ == '__main__':
    try:
        # Print startup information
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        logger.info("Starting Flask application...")
        
        port = int(os.environ.get('PORT', 5000))
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True,
            use_reloader=False  # Disable reloader to prevent duplicate model loading
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise