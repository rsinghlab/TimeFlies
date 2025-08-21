# TimeFlies 🕰️🧬

Machine Learning framework for analyzing aging patterns in Drosophila single-cell RNA sequencing data.

## Features

- **Multi-Project Support**: Supports aging and Alzheimer's disease research
- **Deep Learning Models**: CNN, MLP, and traditional ML approaches
- **SHAP Interpretability**: Model explanation and feature importance
- **Batch Correction**: scVI integration for batch effect removal
- **Comprehensive Testing**: Automated test suite with real data sampling
- **GPU Acceleration**: CUDA support for faster training

## Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd TimeFlies

# Setup development environment (creates .venv)
bash setup_dev_env.sh

# Activate environment
source activate.sh
```

### 2. Verify Installation

```bash
python run_timeflies.py verify
```

### 3. Add Your Data

Place your H5AD files in the correct structure:
```
data/
├── fruitfly_aging/head/
│   ├── drosophila_head_aging_original.h5ad
│   ├── drosophila_head_aging_train.h5ad      # (optional)
│   └── drosophila_head_aging_eval.h5ad       # (optional)
└── fruitfly_alzheimers/head/
    └── drosophila_head_alzheimers_original.h5ad
```

### 4. Complete Workflow

```bash
# Create test fixtures from your data
python run_timeflies.py create-test-data

# Create train/eval splits
python run_timeflies.py setup

# Train model
python run_timeflies.py train

# Evaluate with SHAP interpretation
python run_timeflies.py evaluate
```

## Project Configuration

Active project is set in `configs/default.yaml`:
```yaml
project: fruitfly_aging  # Change to: fruitfly_alzheimers
```

Project-specific configurations in:
- `configs/fruitfly_aging/default.yaml`
- `configs/fruitfly_alzheimers/default.yaml`

## Commands

| Command | Description |
|---------|-------------|
| `verify` | Verify environment and setup |
| `create-test-data` | Generate test fixtures from real data |
| `setup` | Create train/evaluation data splits |
| `train` | Train models with current project settings |
| `evaluate` | Evaluate models with SHAP interpretation |
| `test` | Run automated test suite |
| `batch-correct` | Apply scVI batch correction |

## Batch Correction (Optional)

For batch effect correction using scVI-tools:

```bash
# Setup creates both environments automatically
bash setup_dev_env.sh  # Choose 'y' for batch environment

# Use batch correction
source activate_batch.sh
python run_timeflies.py batch-correct
source activate.sh  # Return to main environment
```

## Testing

```bash
# Run all tests
python run_timeflies.py test

# Run specific test types
python run_timeflies.py test unit
python run_timeflies.py test --fast
python run_timeflies.py test --coverage
```

## Generated Outputs

```
outputs/
├── fruitfly_aging/
│   ├── models/           # Trained models
│   └── results/          # Evaluation results, plots
└── fruitfly_alzheimers/
    ├── models/
    └── results/

tests/fixtures/
├── fruitfly_aging/       # Test data fixtures
│   ├── test_data_head.h5ad
│   └── test_data_head_stats.json
└── fruitfly_alzheimers/
```

## Requirements

- **Python**: 3.10+ (3.12 recommended)
- **GPU**: NVIDIA GPU with CUDA (optional, falls back to CPU)
- **Memory**: 8GB+ RAM recommended
- **Dependencies**: Automatically installed via setup script

## Architecture

```
src/
├── projects/                    # Project-specific code
│   ├── fruitfly_aging/         # Aging project
│   └── fruitfly_alzheimers/    # Alzheimer's project
├── shared/                      # Shared components
│   ├── cli/                    # Command-line interface
│   ├── core/                   # Configuration management
│   ├── data/                   # Data loading utilities
│   ├── models/                 # Model architectures
│   └── utils/                  # Utility functions
configs/                         # Configuration files
tests/                          # Test suite
```

## Contributing

1. Ensure all tests pass: `python run_timeflies.py test`
2. Verify setup works: `python run_timeflies.py verify`
3. Follow existing code patterns and documentation

## License

[Add your license information here]

## Citation

[Add citation information here]