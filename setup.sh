#!/bin/bash


echo "======================================================================"
echo "YOLOv5 TFLite Optimization - Setup Script"
echo "======================================================================"
echo "======================================================================"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p models
mkdir -p data/coco_val2017_subset
mkdir -p results/plots
mkdir -p scripts

echo "✓ Directories created"
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "  Python version: $python_version"

required_version="3.8"
if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then 
    echo "  ✓ Python version is sufficient"
else
    echo "  ✗ Python 3.8+ required"
    exit 1
fi
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "  ✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
echo "  ✓ pip upgraded"
echo ""

# Install requirements
echo "Installing requirements..."
echo "  This may take several minutes..."
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "  ✓ Requirements installed successfully"
else
    echo "  ✗ Error installing requirements"
    exit 1
fi
echo ""

# Verify installations
echo "Verifying installations..."
python3 -c "import tensorflow as tf; print(f'  TensorFlow: {tf.__version__}')"
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python3 -c "import numpy as np; print(f'  NumPy: {np.__version__}')"
python3 -c "import cv2; print(f'  OpenCV: {cv2.__version__}')"
echo "  ✓ All packages verified"
echo ""

echo "======================================================================"
echo "Setup completed successfully!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Activate the virtual environment (if not already active):"
echo "     source venv/bin/activate"
echo ""
echo "  2. Run the optimization pipeline:"
echo "     bash run_all.sh"
echo ""
echo "  Or run scripts individually:"
echo "     python scripts/1_download_model.py"
echo "     python scripts/2_convert_to_tensorflow.py"
echo "     python scripts/3_convert_to_tflite.py"
echo "     python scripts/4_benchmark_latency.py"
echo "     python scripts/5_evaluate_accuracy.py"
echo "     python scripts/6_compare_models.py"
echo "     python scripts/7_visualize_results.py"
echo "     python scripts/8_profile_model.py"
echo ""
echo "======================================================================"
