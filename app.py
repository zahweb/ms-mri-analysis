import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
import logging

logging.getLogger('werkzeug').setLevel(logging.ERROR)

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def home():
    return render_template('index_advanced.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'message': 'MS MRI Analysis Server is running',
        'python_version': '3.9'
    })

@app.route('/test-tda')
def test_tda():
    try:
        from gtda.homology import VietorisRipsPersistence
        return jsonify({'tda_status': '✅ TDA is working!'})
    except ImportError as e:
        return jsonify({'tda_status': f'❌ TDA failed: {e}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Server starting on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
