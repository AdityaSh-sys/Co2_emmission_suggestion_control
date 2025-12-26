#!/bin/bash

echo "🔧 Setting up environment..."

# Create virtual environment
python3 -m venv .venv
source ttsenv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ All dependencies installed. Activate venv using: source env/bin/activate"
