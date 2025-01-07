from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
import os

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_recommendations import RecommendationGenerator

app = Flask(__name__)

# Updated CORS configuration with additional options
CORS(app, resources={
    r"/api/*": {
        "origins": ["https://lhydra.com", "http://localhost:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "expose_headers": ["Content-Range", "X-Content-Range"],
        "supports_credentials": True
    }
})

# Setup paths relative to new project structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(ROOT_DIR, 'checkpoints', 'best_model.pth')
DATA_PATH = os.path.join(ROOT_DIR, 'data', 'test_data.csv')
ENCODERS_PATH = os.path.join(ROOT_DIR, 'data', 'data_encoders.pt')

print(f"Root directory: {ROOT_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Data path: {DATA_PATH}")
print(f"Encoders path: {ENCODERS_PATH}")

# Initialize model and data (done once at startup)
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(f"Encoders file not found at {ENCODERS_PATH}")
    
    catalog_data = pd.read_csv(DATA_PATH)
    recommender = RecommendationGenerator(
        model_path=MODEL_PATH,
        catalog_data=catalog_data,
        encoders_path=ENCODERS_PATH
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.after_request
def after_request(response):
    """Add headers to every response."""
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:3000')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response

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

@app.route('/api/recommendations', methods=['POST', 'OPTIONS'])
def get_recommendations():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.json
        user_info = {
            'user_id': data['user_id'],
            'age': int(data['age']),
            'gender': data['gender'],
            'genre': data['genre'],
            'music': data['music']
        }
        
        recommendations = recommender.generate_recommendations(user_info, n_recommendations=5)
        
        return jsonify({
            'status': 'success',
            'recommendations': recommendations.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    # Bind to the dynamic port for production platforms like Render
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
