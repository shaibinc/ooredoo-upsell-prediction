#!/bin/bash
set -e

# Azure App Service startup script for Ooredoo Upsell Predictor V2
echo "[STARTUP] Starting Ooredoo Upsell Predictor V2 deployment..."

# Set environment variables
export PYTHONUNBUFFERED=1
export FLASK_APP=app.py
export FLASK_ENV=production
export PYTHONPATH=/home/site/wwwroot

# Find the app directory (Azure extracts to different locations)
# Check if we're in the extracted directory already
if [ -f "requirements.txt" ]; then
    APP_DIR="$(pwd)"
else
    # Try common Azure paths
    APP_DIR="/home/site/wwwroot"
    if [ ! -f "$APP_DIR/requirements.txt" ]; then
        # Look for extracted directory
        EXTRACTED_DIR=$(find /tmp -maxdepth 1 -name "8ddb*" -type d 2>/dev/null | head -1)
        if [ -n "$EXTRACTED_DIR" ] && [ -f "$EXTRACTED_DIR/requirements.txt" ]; then
            APP_DIR="$EXTRACTED_DIR"
        fi
    fi
fi

# Navigate to app directory
echo "[STARTUP] Using app directory: $APP_DIR"
cd "$APP_DIR"

# List files for debugging
echo "[STARTUP] Files in app directory:"
ls -la

# Check Python version
echo "[STARTUP] Python version:"
python --version

# Upgrade pip
echo "[STARTUP] Upgrading pip..."
python -m pip install --upgrade pip

# Install dependencies with verbose output
echo "[STARTUP] Installing dependencies..."
if [ -f "requirements.txt" ]; then
    echo "[STARTUP] Installing from requirements.txt"
    python -m pip install -r requirements.txt --verbose --no-cache-dir
else
    echo "[STARTUP] ERROR: requirements.txt not found in $APP_DIR"
    echo "[STARTUP] Available files:"
    find . -name "*.txt" -o -name "*.py" | head -10
    exit 1
fi

# Check if critical packages are installed
echo "[STARTUP] Verifying critical packages..."
python -c "import flask; print('Flask version:', flask.__version__)"
python -c "import pandas; print('Pandas version:', pandas.__version__)"
python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
python -c "import openai; print('OpenAI version:', openai.__version__)"

# Check if pyodbc is available for Azure SQL
echo "[STARTUP] Checking pyodbc availability..."
python -c "import pyodbc; print('pyodbc available for Azure SQL')" || echo "[STARTUP] Warning: pyodbc not available, will use SQLite fallback"

# Set proper permissions
echo "[STARTUP] Setting permissions..."
chmod +x "$APP_DIR/app.py"

# Create logs directory if it doesn't exist
mkdir -p /home/LogFiles

# Ensure we're in the correct directory
echo "[STARTUP] Final working directory: $(pwd)"
echo "[STARTUP] App directory: $APP_DIR"
cd "$APP_DIR"

# Start the application with Gunicorn
echo "[STARTUP] Starting Gunicorn server..."
echo "[STARTUP] Port: ${PORT:-8000}"
echo "[STARTUP] Workers: 4"
echo "[STARTUP] Timeout: 120 seconds"

# Start Gunicorn with production settings
exec gunicorn \
    --bind 0.0.0.0:${PORT:-8000} \
    --workers 4 \
    --worker-class sync \
    --timeout 120 \
    --keep-alive 2 \
    --max-requests 1000 \
    --max-requests-jitter 100 \
    --preload \
    --access-logfile /home/LogFiles/access.log \
    --error-logfile /home/LogFiles/error.log \
    --log-level info \
    --capture-output \
    app:app