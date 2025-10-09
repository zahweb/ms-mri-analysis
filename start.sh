#!/bin/bash
echo "🚀 Starting server with Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT app:app \
  --timeout 300 \
  --workers 1 \
  --worker-class sync \
  --worker-connections 1000 \
  --max-requests 100 \
  --max-requests-jitter 20
