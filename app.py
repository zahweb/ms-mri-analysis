import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
import logging
import sys

logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

print(f"🚀 Python version: {sys.version}")

@app.route('/')
def home():
    return "🚀 MS MRI Analysis Server is Running! Visit /health for status"

@app.route('/health')
def health():
    try:
        # Test basic imports
        import numpy as np
        import tensorflow as tf
        
        # Test TDA
        from gtda.homology import VietorisRipsPersistence
        
        return jsonify({
            'status': 'healthy ✅',
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'tensorflow_version': tf.__version__,
            'tda_status': '✅ TDA is working!'
        })
        
    except ImportError as e:
        return jsonify({
            'status': 'partial ❌',
            'python_version': sys.version,
            'error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
