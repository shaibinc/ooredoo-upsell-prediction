#!/bin/bash

# Azure App Service startup script for Flask application
echo "Starting Ooredoo Upsell Prediction App..."

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH=/home/site/wwwroot

# Create necessary directories
mkdir -p /home/site/wwwroot/logs

# Initialize database if it doesn't exist
cd /home/site/wwwroot
echo "Attempting to initialize database..."
python -c "import sys; sys.path.insert(0, '.'); from app import init_database; init_database()" 2>/dev/null || echo "Database initialization failed, will retry on first request"

# Start the Flask application
echo "Starting Flask application..."
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --preload app:app