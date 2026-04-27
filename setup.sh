#!/bin/bash
# =============================================================================
# Thesis Project: BLAIR Verification & Extended Recommender Benchmarking
# Setup script for GPU server
# =============================================================================

set -e

# Detect CUDA version
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA Version detected:"
    nvidia-smi --query-gpu=cuda_version --format=csv,noheader | head -1
else
    echo "WARNING: nvidia-smi not found. Ensure CUDA is installed."
fi

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip wheel setuptools

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118

# Install all dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Install RecBole from source (latest version)
echo "Installing RecBole..."
pip install recbole

# Clone BLAIR reference implementation
echo "Cloning BLAIR reference repo..."
if [ ! -d "blair_reference" ]; then
    git clone https://github.com/hyp1231/AmazonReviews2023.git blair_reference
fi

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed data/cache
mkdir -p checkpoints/blair checkpoints/deepseek checkpoints/claude checkpoints/custom
mkdir -p results/sequential_recommendation results/product_search results/complex_search
mkdir -p logs

# Setup environment variables
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Add in your own API key(s)."
fi

# Download BLAIR checkpoints
echo "Downloading BLAIR checkpoints..."
python -c "
from transformers import AutoModel, AutoTokenizer
print('Downloading blair-roberta-base...')
AutoTokenizer.from_pretrained('hyp1231/blair-roberta-base')
AutoModel.from_pretrained('hyp1231/blair-roberta-base')
print('Downloading blair-roberta-large...')
AutoTokenizer.from_pretrained('hyp1231/blair-roberta-large')
AutoModel.from_pretrained('hyp1231/blair-roberta-large')
print('Done!')
"