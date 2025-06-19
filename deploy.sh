#!/bin/bash

# Azure Custom Deployment Script for Ooredoo Upsell App
echo "Starting custom deployment for Ooredoo Upsell Prediction App..."

# Exit on any error
set -e

# Deployment settings
DEPLOYMENT_SOURCE=${DEPLOYMENT_SOURCE:-$PWD}
DEPLOYMENT_TARGET=${DEPLOYMENT_TARGET:-/home/site/wwwroot}
NEXT_MANIFEST_PATH=${NEXT_MANIFEST_PATH:-$DEPLOYMENT_TARGET/manifest}
PREVIOUS_MANIFEST_PATH=${PREVIOUS_MANIFEST_PATH:-$NEXT_MANIFEST_PATH}
KUDU_SYNC_CMD=${KUDU_SYNC_CMD:-/opt/Kudu/bin/kudusync}
PYTHON_RUNTIME=${PYTHON_RUNTIME:-python-3.9}
PYTHON_VER=${PYTHON_VER:-3.9}
PYTHON_EXE=$DEPLOYMENT_TARGET/antenv/bin/python
PIP_CMD="$PYTHON_EXE -m pip"

echo "Deployment source: $DEPLOYMENT_SOURCE"
echo "Deployment target: $DEPLOYMENT_TARGET"

# 1. KuduSync
echo "Step 1: Syncing files..."
if [[ "$IN_PLACE_DEPLOYMENT" -ne "1" ]]; then
  "$KUDU_SYNC_CMD" -v 50 -f "$DEPLOYMENT_SOURCE" -t "$DEPLOYMENT_TARGET" -n "$NEXT_MANIFEST_PATH" -p "$PREVIOUS_MANIFEST_PATH" -i ".git;.hg;.deployment;deploy.sh"
  exitWithMessageOnError "Kudu Sync failed"
fi

# 2. Create virtual environment
echo "Step 2: Creating virtual environment..."
cd "$DEPLOYMENT_TARGET"
if [ ! -d "antenv" ]; then
  python$PYTHON_VER -m venv antenv
  exitWithMessageOnError "Failed to create virtual environment"
fi

# 3. Activate virtual environment
echo "Step 3: Activating virtual environment..."
source antenv/bin/activate
exitWithMessageOnError "Failed to activate virtual environment"

# 4. Install packages
echo "Step 4: Installing Python packages..."
if [ -e "$DEPLOYMENT_TARGET/requirements.txt" ]; then
  $PIP_CMD install --upgrade pip
  $PIP_CMD install -r requirements.txt
  exitWithMessageOnError "pip install failed"
fi

# 5. Initialize database
echo "Step 5: Initializing database..."
$PYTHON_EXE -c "from app import init_database; init_database()"
exitWithMessageOnError "Database initialization failed"

# 6. Set permissions
echo "Step 6: Setting file permissions..."
chmod +x startup.sh

echo "Deployment completed successfully!"

# Helper function
exitWithMessageOnError () {
  if [ ! $? -eq 0 ]; then
    echo "An error has occurred during web site deployment."
    echo $1
    exit 1
  fi
}