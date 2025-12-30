from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add the current directory to python path so we can import modules
sys.path.append(os.getcwd())

try:
    from frontends.source.cuda_frontend import translate_cuda_source
except ImportError as e:
    print(f"Error importing translation module: {e}")
    # Fallback for dev/testing if modules aren't fully set up
    def translate_cuda_source(source, target="hexagon"):
        return f"// Error: Could not import translator.\n// {str(e)}\n\n// Mock output:\n" + source.replace("__global__", "kernel")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    if not data or 'source' not in data:
        return jsonify({'error': 'No source code provided'}), 400
    
    source_code = data['source']
    try:
        # Perform the translation
        translated_code = translate_cuda_source(source_code, target="hexagon")
        return jsonify({'translated': translated_code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server on http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
