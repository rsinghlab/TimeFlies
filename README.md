# TimeFlies 🧬

**Machine Learning for Aging Analysis**

A modern machine learning framework for analyzing aging patterns in Drosophila single-cell RNA sequencing data using deep learning with comprehensive visualization and SHAP interpretability.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone and setup environment
git clone <repository>
cd TimeFlies
bash setup_dev_env.sh

# Activate environment  
source activate.sh
```

### 2. Data Preparation
Place your H5AD files in the data directory:
```
data/
├── fruitfly_aging/head/
│   └── drosophila_head_aging_original.h5ad
└── fruitfly_alzheimers/head/
    └── drosophila_head_alzheimers_original.h5ad
```

### 3. Complete Workflow
```bash
# Create test fixtures for development
python run_timeflies.py create-test-data

# Setup train/eval splits for all projects  
python run_timeflies.py setup

# Verify everything is ready
python run_timeflies.py verify

# Train models
python run_timeflies.py train

# Evaluate with SHAP interpretation (optional)
python run_timeflies.py evaluate
```

## 📋 Core Commands

| Command | Description |
|---------|-------------|
| `create-test-data` | Create test fixtures from real data |
| `setup` | Create stratified train/eval splits for all projects |
| `verify` | Verify environment and data setup |
| `train` | Train deep learning models |
| `evaluate` | Evaluate models with SHAP interpretation |
| `test` | Run test suite |

> 📖 **Detailed CLI reference**: See [docs/COMMANDS.md](docs/COMMANDS.md) for all options and examples

## 🔧 Configuration

### Switch Projects
Edit `configs/default.yaml`:
```yaml
project: fruitfly_aging  # or fruitfly_alzheimers
```

### Setup Parameters
Configure splits in `configs/setup.yaml`:
```yaml
data:
  train_test_split:
    split_size: 5000  # cells in eval set
general:
  random_state: 42    # for reproducible splits
```

## 🧬 Data Setup Features

### Multi-Project Support
- Automatically detects all projects in `data/` directory
- Processes both original and batch-corrected data
- Creates project-specific output directories

### Advanced Stratification
- **Primary**: Disease/age status from project config
- **Secondary**: Sex (sex/Sex/gender/Gender columns)
- **Tertiary**: Cell type (cell_type/celltype/cluster columns)

### Batch Correction Support
```bash
# Optional: Setup batch correction environment
bash setup_dev_env.sh  # choose 'y' for batch environment

# Activate batch environment
source activate_batch.sh

# Run batch correction (creates *_original_batch.h5ad)
python run_timeflies.py batch-correct

# Return to main environment
source activate.sh

# Re-run setup to create batch splits
python run_timeflies.py setup
```

## 📊 File Structure

### Data Organization
```
data/
├── fruitfly_aging/head/
│   ├── drosophila_head_aging_original.h5ad      # Source
│   ├── drosophila_head_aging_original_batch.h5ad # Batch corrected
│   ├── drosophila_head_aging_train.h5ad         # Training split
│   ├── drosophila_head_aging_eval.h5ad          # Evaluation split
│   ├── drosophila_head_aging_train_batch.h5ad   # Batch training split
│   └── drosophila_head_aging_eval_batch.h5ad    # Batch evaluation split
└── fruitfly_alzheimers/head/
    └── [similar structure]
```

### Output Organization  
```
outputs/
├── fruitfly_aging/
│   ├── models/     # Trained models
│   ├── results/    # SHAP analysis, plots
│   └── logs/       # Training logs
└── fruitfly_alzheimers/
    └── [similar structure]
```

## 🧪 Development

### Test Suite
```bash
# Run all tests
python run_timeflies.py test

# Run specific test types
python run_timeflies.py test --type unit
python run_timeflies.py test --type integration
python run_timeflies.py test --fast
```

### Project Structure
```
src/
├── shared/           # Core framework
│   ├── cli/         # Command line interface
│   ├── core/        # Configuration management
│   ├── data/        # Data loading and setup
│   ├── models/      # Model architectures
│   └── utils/       # Utilities
└── projects/        # Project-specific code
    ├── fruitfly_aging/
    └── fruitfly_alzheimers/
```

## 🎯 Key Features

- **🔄 Automated Setup**: One command sets up all projects
- **🧬 Smart Stratification**: Multi-level stratified splitting
- **🔒 Safe Operations**: Never overwrites existing data
- **📊 Batch Support**: Handles both original and batch-corrected data  
- **🎨 Clean Output**: Organized with clear progress indicators
- **🧪 Comprehensive Testing**: Unit, integration, and system tests
- **📈 SHAP Analysis**: Model interpretability with visualizations

## 🔬 Supported Projects

- **Fruitfly Aging**: Age-related gene expression analysis
- **Fruitfly Alzheimers**: Disease condition classification
- **Extensible**: Easy to add new projects

## ⚙️ Requirements

- Python 3.12+
- GPU support (optional, CPU fallback available)
- 8GB+ RAM recommended for large datasets

## 📄 License

See `LICENSE` file for details.

## Future Improvements

- Convert to Pytorch
- Combine batch correction and main env into one

---