#!/bin/bash
# Installation and Quick Test Script for Hybrid Model

set -e  # Exit on error

echo "=========================================="
echo "Hybrid TDA-Text Model - Installation"
echo "=========================================="

# Activate virtual environment
cd /mnt/d/Verilogos
source test-env/bin/activate

echo ""
echo "Step 1: Installing PyTorch (CPU version)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Step 2: Installing Transformers..."
pip install transformers

echo ""
echo "Step 3: Installing visualization libraries..."
pip install matplotlib seaborn

echo ""
echo "Step 4: Installing tqdm..."
pip install tqdm

echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="

echo ""
echo "Verifying installation..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"
python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib: {matplotlib.__version__}')"

echo ""
echo "=========================================="
echo "Running Quick Test (50 events)"
echo "=========================================="

cd /mnt/d/Verilogos/hybrid_model

python main_experiment.py \
    --max_events 50 \
    --batch_size 8 \
    --num_epochs 3 \
    --n_workers_tda 2 \
    --modes tda_only text_only

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Check results in: /mnt/d/Verilogos/hybrid_model/results/"
echo ""
