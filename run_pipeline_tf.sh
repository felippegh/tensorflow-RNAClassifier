#!/bin/bash

# TensorFlow 1.x pipeline runner script
# Matches the original workflow but with modern structure

set -e  # Exit on error

echo "=========================================="
echo "RNA Classification - TensorFlow 1.x Pipeline"
echo "=========================================="

# Default values
MAX_SEQUENCES=${1:-1000}
CV_FOLDS=${2:-10}
LEARNING_RATE=${3:-0.01}
L2_REG=${4:-0.1}
EPOCHS=${5:-100}

echo ""
echo "Configuration:"
echo "  Max sequences per class: $MAX_SEQUENCES"
echo "  Cross-validation folds: $CV_FOLDS"
echo "  Learning rate: $LEARNING_RATE"
echo "  L2 regularization: $L2_REG"
echo "  Training epochs: $EPOCHS"
echo ""

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Check for virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: No virtual environment detected"
    echo "It's recommended to use a virtual environment:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Install dependencies if needed
echo "Checking TensorFlow 1.x dependencies..."
python3 -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)" 2>/dev/null || {
    echo "Installing TensorFlow 1.x and dependencies..."
    pip install -r requirements.txt
}

# Run the TensorFlow pipeline
echo ""
echo "Starting TensorFlow 1.x pipeline..."
echo "=========================================="

python3 main.py pipeline \
    --max-sequences "$MAX_SEQUENCES" \
    --cv-folds "$CV_FOLDS" \
    --learning-rate "$LEARNING_RATE" \
    --l2-reg "$L2_REG" \
    --epochs "$EPOCHS" \
    --log-level INFO

echo ""
echo "=========================================="
echo "TensorFlow pipeline completed successfully!"
echo ""
echo "You can also run the original-style workflow:"
echo "  python3 run_original_tf.py $MAX_SEQUENCES"
echo ""
echo "Check the results/ directory for metrics"
echo "Check the models/ directory for saved models"
echo "=========================================="