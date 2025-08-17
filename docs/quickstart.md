# TimeFlies Quick Start Guide

Get up and running with TimeFlies in minutes.

## Prerequisites

- Python 3.11+ (3.12 recommended)
- CUDA-compatible GPU (optional, CPU supported)
- 16GB+ RAM for large datasets

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/rsinghlab/TimeFlies.git
cd TimeFlies
```

### 2. Install Dependencies
```bash
# Create environment
conda create -n timeflies python=3.12
conda activate timeflies

# Install dependencies (choose your platform)
pip install -r requirements/linux/requirements.txt        # Linux
pip install -r requirements/macOS/requirements.txt        # macOS  
pip install -r requirements/windows/requirements.txt      # Windows

# Install TimeFlies in development mode
pip install -e .
```

### 3. Verify Installation
```bash
python -c "from src.timeflies.core.config_manager import ConfigManager; print('✅ Installation successful')"
```

## Basic Usage

### Step 1: Setup Data (Required First)
```bash
# This MUST be run first to create train/test splits
python run_setup.py
```

### Step 2: Train a Model
```bash
# Basic training with default settings
python run_timeflies.py train --tissue head --model cnn --target age

# With custom configuration
python run_timeflies.py train --config configs/my_config.yaml
```

### Step 3: Evaluate Results
Check the outputs in:
- `outputs/models/` - Trained models
- `outputs/results/` - Analysis results and visualizations
- `outputs/logs/` - Execution logs

## Common Workflows

### Age Prediction (Default)
```bash
python run_setup.py
python run_timeflies.py train --tissue head --model cnn --target age
```

### Sex Classification
```bash
python run_setup.py --config configs/sex_config.yaml
python run_timeflies.py train --tissue head --model mlp --target sex
```

### With Batch Correction
```bash
python run_setup.py --config configs/batch_config.yaml
python run_timeflies.py train --batch-correction --tissue head --model cnn
```

### Skip Visualization (Faster)
```bash
python run_timeflies.py train --tissue head --model cnn --target age
# Edit configs/default.yaml and set run_visualization: false
```

## Configuration

All settings are controlled via YAML files in `configs/`. Key options:

```yaml
# Enable/disable features
feature_importance:
  run_interpreter: true     # SHAP analysis
  run_visualization: true   # Generate plots
  load_SHAP: false         # Load existing SHAP

# Model loading
data_processing:
  model_management:
    load_model: false       # Load existing model

# CNN architecture  
model:
  cnn:
    filters: [32]           # Single layer
    dense_units: [128]      # Dense layer size
    dropout_rate: 0.5
```

## Data Structure

Place your `.h5ad` files in:
```
data/raw/h5ad/
├── head/
│   └── fly_original.h5ad
├── body/
│   └── fly_original.h5ad
└── all/
    └── fly_original.h5ad
```

## Next Steps

- Read the [User Guide](user_guide.md) for detailed usage
- Check [Configuration Reference](configuration.md) for all options
- Review [Examples](examples.md) for advanced use cases
- See [CLI Reference](cli_reference.md) for all command options

## Troubleshooting

**Import Errors**: Make sure you're in the TimeFlies root directory and have activated the conda environment.

**Memory Issues**: Reduce batch size in config: `model.training.batch_size: 16`

**CUDA Issues**: Add `--cpu` flag or set `device.processor: "CPU"` in config.

**Setup Issues**: Ensure `run_setup.py` completed successfully before training.