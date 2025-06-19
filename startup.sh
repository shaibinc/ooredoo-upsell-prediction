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
python -c "from app import init_database; init_database()"

# Start the Flask application
echo "Starting Flask application on port 8000..."
gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 120 app:app