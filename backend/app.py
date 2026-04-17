from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import tempfile
from werkzeug.utils import secure_filename
import sys

# Import the predictor from module 4
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from backend.module_4_real_time_detection import XMLParserPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Configuration
UPLOAD_FOLDER = tempfile.gettempdir()
ALLOWED_EXTENSIONS = {'xml'}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Initialize predictor once at startup
predictor = None

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.before_request
def initialize_predictor():
    """Initialize predictor on first request"""
    global predictor
    if predictor is None:
        try:
            predictor = XMLParserPredictor(model_dir=".")
            print("✅ Predictor initialized successfully!")
        except Exception as e:
            print(f"⚠️ Error initializing predictor: {e}")
            print("📄 Running in rule-based mode only")
            predictor = XMLParserPredictor(model_dir=".")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': predictor is not None and predictor.svm_classifier is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Validate request
        if 'xml_file' not in request.files:
            return jsonify({'error': 'No XML file provided'}), 400
        
        if 'cpu_cores' not in request.form:
            return jsonify({'error': 'CPU cores not specified'}), 400
        
        file = request.files['xml_file']
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only XML files are allowed'}), 400
        
        # Get CPU cores
        try:
            cpu_cores = int(request.form['cpu_cores'])
            if cpu_cores not in [1, 2, 4, 6, 8, 10, 12, 14, 16]:
                return jsonify({'error': 'Invalid CPU cores value'}), 400
        except ValueError:
            return jsonify({'error': 'CPU cores must be an integer'}), 400
        
        # Save file temporarily
        filename = secure_filename(file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(temp_path)
        
        try:
            # Make prediction
            result = predictor.predict_from_xml(temp_path, cpu_cores)
            
            # Get parser code
            parser_code = predictor.get_parser_code(result['algorithm'])
            
            # Prepare response
            response = {
                'algorithm': result['algorithm'],
                'confidence': float(result['confidence']),
                'method': result['method'],
                'reason': result['reason'],
                'extracted_features': result['extracted_features'],
                'parser_code': parser_code
            }
            
            # Add ML prediction details if available
            if 'ml_prediction' in result:
                response['ml_prediction'] = result['ml_prediction']
                response['ml_confidence'] = float(result['ml_confidence'])
                response['rule_prediction'] = result['rule_prediction']
                response['rule_reason'] = result['rule_reason']
                response['all_probabilities'] = result['all_probabilities']
            
            return jsonify(response), 200
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file: {e}")
    
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/algorithms', methods=['GET'])
def get_algorithms():
    """Get list of available algorithms"""
    algorithms = [
        {
            'name': 'DOM',
            'description': 'Best for small files (0-4MB)',
            'color': '#3b82f6'
        },
        {
            'name': 'JDOM',
            'description': 'Good for medium files (4-25MB)',
            'color': '#10b981'
        },
        {
            'name': 'SAX',
            'description': 'Excellent for large files (25-100MB)',
            'color': '#a855f7'
        },
        {
            'name': 'StAX',
            'description': 'Optimal for very large files (100+MB)',
            'color': '#f97316'
        },
        {
            'name': 'PXTG',
            'description': 'Complex XML structures',
            'color': '#ec4899'
        }
    ]
    return jsonify(algorithms), 200

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 200MB'}), 413

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("="*70)
    print("🚀 XML Parser Predictor - Flask Backend")
    print("="*70)
    print("📍 Server running on: http://localhost:5000")
    print("🔧 Models directory: .")
    print("📂 Upload folder:", UPLOAD_FOLDER)
    print("="*70)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)