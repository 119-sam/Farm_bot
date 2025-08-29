#!/bin/bash
echo "Starting AgriBot Flask application on port $PORT"
gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2 --timeout 60