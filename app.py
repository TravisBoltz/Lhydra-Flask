from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import sys
import os

# Add parent directory to path to import from sibling directories
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generate_recommendations import RecommendationGenerator

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": ["https://lhydra.com", "http://localhost:3000"]}})
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000", "https://lhydra.com"]}})

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
        encoders_path=ENCODERS_PATH  # Pass encoders path to RecommendationGenerator
    )
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

@app.route('/api/recommendations', methods=['POST'])
def get_recommendations():
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

@app.after_request
def apply_cors(response):
    response.headers['Access-Control-Allow-Origin'] = 'http://localhost:3000'  
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
