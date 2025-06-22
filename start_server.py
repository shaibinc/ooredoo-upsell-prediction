#!/usr/bin/env python3
"""
Simple server starter to avoid double initialization issues
"""

import sys
import os
from app import app, predictor, model_trained

if __name__ == '__main__':
    print("Starting Ooredoo Upsell Prediction V2 Server...")
    print(f"Model trained: {model_trained}")
    print(f"Predictor available: {predictor is not None}")
    
    if predictor and model_trained:
        print("✅ Application ready with trained model!")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Model not properly initialized")
        sys.exit(1)