#!/bin/bash
echo "🚀 Starting server with Gunicorn..."
gunicorn --bind 0.0.0.0:$PORT app:app --timeout 120 --workers 1
