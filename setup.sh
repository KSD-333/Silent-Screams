#!/bin/bash
# Silent Screams - Unix/Linux/macOS Setup Script
# This script automates the installation process

echo "============================================================"
echo "Silent Screams - Automated Setup"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/5] Python detected"
python3 --version
echo ""

# Create virtual environment
echo "[2/5] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping..."
else
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment"
        exit 1
    fi
    echo "Virtual environment created successfully"
fi
echo ""

# Activate virtual environment
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment"
    exit 1
fi
echo ""

# Upgrade pip
echo "[4/5] Upgrading pip..."
pip install --upgrade pip
echo ""

# Install dependencies
echo "[5/5] Installing dependencies..."
echo "This may take 5-10 minutes..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install dependencies"
    exit 1
fi
echo ""

echo "============================================================"
echo "Setup Complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Run verification: python verify_setup.py"
echo "  2. Start application: streamlit run app.py"
echo ""
echo "The virtual environment is now active."
echo "To deactivate, type: deactivate"
echo ""
