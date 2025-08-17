# TimeFlies Environment Setup

This directory contains environment configurations for the TimeFlies project. The project currently uses a **dual-environment setup**:

- **TensorFlow environment** - Main ML pipeline (CNN, MLP, XGBoost, etc.)
- **PyTorch environment** - Batch correction using scVI

## Environment Files

### All Platforms
- `linux/requirements.txt` - Linux dependencies
- `macOS/requirements.txt` - macOS dependencies  
- `windows/requirements.txt` - Windows dependencies
- `linux/batch_environment.yml` - PyTorch-based scVI batch correction environment

## Quick Start

### 1. Main Pipeline (TensorFlow)
```bash
# Create main environment
conda create -n timeflies python=3.12
conda activate timeflies

# Install dependencies (choose your platform)
pip install -r Requirements/linux/requirements.txt     # Linux
pip install -r Requirements/macOS/requirements.txt     # macOS
pip install -r Requirements/windows/requirements.txt   # Windows

# Install TimeFlies
pip install -e .

# Test installation
python run_timeflies.py test
```

### 2. Batch Correction (PyTorch + scVI)
```bash
# Create batch correction environment
conda env create -f Requirements/linux/batch_environment.yml
conda activate scvi-env

# Test batch correction
python run_timeflies.py batch --train --tissue head
```

## Usage Workflow

### Training Models (TensorFlow Environment)
```bash
conda activate timeflies

# Setup data (required first)
python run_setup.py

# Train models
python run_timeflies.py train --tissue head --model cnn --target age
python run_timeflies.py train --tissue head --model mlp --target age
python run_timeflies.py train --tissue head --model xgboost --target age
```

### Batch Correction (PyTorch Environment)  
```bash
conda activate scvi-env

# Note: Batch correction is integrated into main pipeline
# Use batch correction config files or --batch-correction flag
python run_timeflies.py train --tissue head --model cnn --target age --batch-correction
```

### Complete Workflow
```bash
# Main environment handles everything now
conda activate timeflies

# Setup data splits
python run_setup.py

# Train with batch correction
python run_timeflies.py train --tissue head --model cnn --target age --batch-correction

# Evaluate trained model
python run_timeflies.py evaluate --model-path outputs/models/head_cnn_age/
```

## System Requirements

### Minimum
- **Python**: 3.12+
- **RAM**: 16GB (32GB+ recommended)
- **Storage**: 10GB free space
- **OS**: Linux (Ubuntu 20.04+), macOS 12+, Windows 10+

### Recommended (GPU)
- **NVIDIA GPU**: 8GB+ VRAM (for batch correction)
- **RAM**: 32GB+
- **Storage**: SSD with 50GB+ free space

## Installation Guide

### 1. Prerequisites
```bash
# Ubuntu/Debian
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git

# NVIDIA drivers (for GPU)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### 2. Clone Repository
```bash
git clone https://github.com/rsinghlab/TimeFlies.git
cd TimeFlies
```

### 3. Setup Both Environments
```bash
# Main environment (TensorFlow)
conda create -n timeflies python=3.12
conda activate timeflies
pip install -r Requirements/linux/requirements.txt

# Batch correction environment (PyTorch)
conda env create -f Requirements/linux/batch_environment.yml
```

### 4. Verify Installation
```bash
# Test main pipeline
conda activate timeflies
python run_timeflies.py test

# Test batch correction
conda activate scvi-env
python -c "import scvi; print('scVI version:', scvi.__version__)"
```

## Troubleshooting

### Environment Switching
```bash
# Check current environment
conda info --envs

# Always activate correct environment before running commands
conda activate timeflies      # for main pipeline
conda activate scvi-env       # for batch correction
```

### Common Issues

#### Import Errors
```bash
# Wrong environment activated
conda activate timeflies  # or scvi-env

# Missing packages
pip install --upgrade [package-name]
```

#### GPU Issues (Batch Correction)
```bash
# Check CUDA
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

#### Memory Issues
```bash
# Reduce batch size in configs/default.yaml
model:
  training:
    batch_size: 16  # reduce from 32
```

## Development Notes

### Current Architecture
- **Main Pipeline**: TensorFlow/Keras for CNN, MLP models
- **Batch Correction**: PyTorch/scVI for technical variation removal
- **Traditional ML**: scikit-learn for XGBoost, Random Forest, Logistic Regression

### Future Migration (Planned)
- **Goal**: Unified PyTorch environment
- **Benefits**: Single environment, better ecosystem integration
- **Timeline**: After current refactoring is complete

### Environment Management Tips
```bash
# Export environments for backup
conda env export -n timeflies > timeflies_backup.yml
conda env export -n scvi-env > scvi_backup.yml

# Update environments
pip install --upgrade -r Requirements/linux/requirements.txt
conda env update -f Requirements/linux/batch_environment.yml
```

## Getting Help

- **Installation Issues**: Check troubleshooting section
- **Scientific Questions**: Open GitHub issue
- **Environment Conflicts**: Ensure correct environment is activated

For more details, see the main [README.md](../README.md).