#!/bin/bash

# Azure App Service startup script for Flask application
echo "Starting Ooredoo Upsell Prediction App..."

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH=/home/site/wwwroot

# Create necessary directories
mkdir -p /home/site/wwwroot/logs

# Database will be initialized on first request
cd /home/site/wwwroot
echo "Database will be initialized on first request"

# Start the Flask application
echo "Starting Flask application..."
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 --preload app:app