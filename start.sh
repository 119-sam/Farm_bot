#!/bin/bash
echo "Starting AgriBot Flask application on port $PORT"
exec python app.py --port=$PORT


