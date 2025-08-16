# TimeFlies: Machine Learning for Aging Analysis in Drosophila Single-Cell RNA-seq Data

A comprehensive machine learning pipeline for analyzing aging patterns in Drosophila (fruit fly) single-cell RNA sequencing data. This project provides tools for batch correction, preprocessing, model training, and visualization of gene expression patterns across different ages, sexes, and tissue types.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Data Pipeline](#data-pipeline)
- [Supported Models](#supported-models)
- [Batch Correction](#batch-correction)
- [Analysis and Visualization](#analysis-and-visualization)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

TimeFlies is designed to analyze single-cell RNA sequencing data from Drosophila melanogaster to understand aging patterns at the cellular level. The pipeline supports:

- **Multiple tissue types**: Head, body, or combined analysis
- **Cross-sex analysis**: Train on one sex, test on another
- **Batch correction**: Using scVI for technical variation removal
- **Multiple ML models**: CNN, MLP, XGBoost, Random Forest, Logistic Regression
- **Comprehensive visualization**: UMAP, PCA, feature importance, and statistical analysis

## Features

### Data Processing
- ✅ Flexible data filtering (sex, cell type, tissue type)
- ✅ Gene filtering options (autosomal, sex-linked, lncRNA genes)
- ✅ Highly variable gene selection
- ✅ Stratified train/test splitting
- ✅ Data normalization and scaling

### Batch Correction
- ✅ scVI-based batch correction
- ✅ UMAP visualization before/after correction
- ✅ Batch correction quality metrics

### Machine Learning
- ✅ Multiple model architectures (CNN, MLP, traditional ML)
- ✅ Model checkpointing and loading
- ✅ Cross-validation support
- ✅ Feature importance analysis (SHAP)

### Visualization & Analysis
- ✅ UMAP and PCA plots
- ✅ Gene expression heatmaps
- ✅ Statistical analysis and outlier detection
- ✅ ROC curves and performance metrics
- ✅ Confusion matrices

## Project Structure

```
TimeFlies/
├── data/                           # All input and processed data
│   ├── raw/                        # Raw input data
│   │   ├── h5ad/                   # Original h5ad files
│   │   │   ├── head/
│   │   │   │   ├── fly_eval.h5ad           (uncorrected)
│   │   │   │   └── fly_eval_batch.h5ad     (batch corrected)
│   │   │   ├── body/
│   │   │   │   ├── fly_eval.h5ad           (uncorrected)
│   │   │   │   └── fly_eval_batch.h5ad     (batch corrected)
│   │   │   └── all/
│   │   │       ├── fly_eval.h5ad           (uncorrected)
│   │   │       └── fly_eval_batch.h5ad     (batch corrected)
│   │   └── gene_lists/
│   │       ├── autosomal.csv
│   │       └── sex.csv
│   └── processed/                  # Preprocessed data with organized structure
│       ├── batch_corrected/
│       │   ├── head_cnn_age/
│       │   │   └── all-genes_all-cells_all-sexes/
│       │   └── body_cnn_age/
│       │       └── all-genes_all-cells_all-sexes/
│       └── uncorrected/
│           ├── head_cnn_age/
│           │   ├── all-genes_all-cells_all-sexes/
│           │   ├── no-sex_all-cells_all-sexes/
│           │   ├── only-lnc_epithelial-cell_all-sexes/
│           │   └── all-genes_muscle-cell_all-sexes/
│           ├── body_cnn_age/
│           │   ├── all-genes_all-cells_all-sexes/
│           │   └── only-lnc_all-cells_all-sexes/
│           └── head_logistic_regression_age/
│               ├── all-genes_all-cells_all-sexes/
│               └── only-lnc_all-cells_all-sexes/
├── outputs/                        # All analysis outputs
│   ├── models/                     # Trained models
│   │   ├── batch_corrected/
│   │   │   └── head_cnn_age/
│   │   │       └── all-genes_all-cells_all-sexes/
│   │   │           ├── model.h5
│   │   │           ├── config.yaml
│   │   │           ├── metrics.json
│   │   │           └── history.pkl
│   │   └── uncorrected/
│   ├── results/                    # Analysis results and plots
│   │   ├── batch_corrected/
│   │   │   └── head_cnn_age/
│   │   │       └── all-genes_all-cells_all-sexes/
│   │   │           ├── plots/
│   │   │           │   ├── confusion_matrix.png
│   │   │           │   ├── roc_curve.png
│   │   │           │   └── training_metrics.png
│   │   │           └── Stats.JSON
│   │   └── uncorrected/
│   └── logs/                       # Log files
│       ├── training_2024_08_16.log
│       └── pipeline_debug.log
├── src/timeflies/                  # Modular source code
│   ├── __init__.py
│   ├── core/                       # Core pipeline components
│   │   ├── config_manager.py       # Configuration management
│   │   ├── pipeline_manager.py     # Pipeline orchestration
│   │   └── config.py               # Configuration schemas
│   ├── data/                       # Data handling
│   │   ├── loaders.py              # Data loading utilities
│   │   └── preprocessing/
│   │       ├── data_processor.py   # Main preprocessing
│   │       ├── gene_filter.py      # Gene filtering logic
│   │       └── batch_correction.py # Batch correction
│   ├── models/                     # ML models
│   │   ├── model_factory.py        # Model creation
│   │   └── model.py                # Base model classes
│   ├── analysis/                   # Analysis tools
│   │   ├── eda.py                  # Exploratory data analysis
│   │   └── visuals.py              # Visualization functions
│   ├── evaluation/
│   │   └── interpreter.py          # Model interpretation (SHAP)
│   └── utils/                      # Utilities
│       ├── path_manager.py         # Path management
│       ├── cli_parser.py           # Command line interface
│       ├── logging_config.py       # Logging setup
│       ├── gpu_handler.py          # GPU configuration
│       ├── constants.py            # Constants
│       └── exceptions.py           # Custom exceptions
├── configs/                        # Configuration files
│   └── default.yaml                # Default pipeline configuration
├── scripts/                        # Utility scripts
├── tests/                          # Test suite
├── notebooks/                      # Analysis notebooks
│   └── analysis.ipynb              # Interactive analysis
├── Requirements/                   # Environment specifications
│   ├── linux/
│   │   ├── requirements.txt        # Linux dependencies
│   │   └── batch_environment.yml   # Linux batch environment
│   ├── macOS/
│   │   └── mac_gpu.yml             # macOS environment
│   └── windows/
│       └── Windows_GPU.yml         # Windows environment
├── run_timeflies.py                # Main entry point
├── requirements_dev.txt            # Development dependencies
├── pyproject.toml                  # Python project configuration
└── README.md                       # This file
```

## Folder Naming Convention

TimeFlies uses a consistent, hierarchical naming convention for organized data management:

### Two-Level Organization
- **Level 1**: Experiment Type (`tissue_model_encoding`)
- **Level 2**: Configuration Details (`method_cells_sexes`)

### Naming Rules
- **Between main parts**: `_` (underscores)
- **Within compound terms**: `-` (hyphens)

### Examples
```
# Level 1: Experiment Type
head_cnn_age/           # Head tissue, CNN model, age prediction
body_mlp_tissue/        # Body tissue, MLP model, tissue classification
head_logistic_regression_age/  # Head tissue, logistic regression, age prediction

# Level 2: Configuration Details  
all-genes_all-cells_all-sexes/     # All genes, all cell types, all sexes
no-sex_muscle-cell_male/           # No sex genes, muscle cells only, males only
only-lnc_epithelial-cell_female/   # Only lncRNA genes, epithelial cells, females
```

### Gene Methods
- `all-genes`: Complete gene set
- `no-sex`: Remove sex-linked genes
- `no-autosomal`: Remove autosomal genes  
- `only-lnc`: Only lncRNA genes
- `hvg`: Highly variable genes
- `balanced`: Balanced gene selection

### Cell Types
- `all-cells`: All cell types
- `muscle-cell`: Muscle cells only
- `epithelial-cell`: Epithelial cells only
- `cns-neuron`: CNS neurons only

### Sex Types
- `all-sexes`: Both male and female
- `male`: Male samples only
- `female`: Female samples only

## Installation

### Prerequisites
- Python 3.12+
- CUDA-compatible GPU (recommended)
- Git
- Conda (Anaconda or Miniconda)

### Step-by-Step Installation

#### 1. Clone Repository
```bash
git clone https://github.com/rsinghlab/TimeFlies.git
cd TimeFlies
```

#### 2. Set Up Environment
```bash
# Create and activate environment
conda create -n timeflies python=3.12
conda activate timeflies

# Install dependencies
pip install -r requirements_dev.txt
```

#### 3. Verify Installation
```bash
python run_timeflies.py test
```

## Configuration

TimeFlies uses YAML configuration files for flexible pipeline setup. The main configuration is in `configs/default.yaml`.

### Basic Settings
```yaml
# General settings
general:
  project_name: "TimeFlies"
  random_state: 42

# Data parameters
data:
  tissue: "head"                    # 'head', 'body', 'all'
  model_type: "CNN"                 # 'CNN', 'MLP', 'logistic', etc.
  encoding_variable: "age"          # 'age', 'sex', 'tissue'
  cell_type: "all"                  # Cell type filter
  sex_type: "all"                   # 'all', 'male', 'female'
  
  # Batch correction
  batch_correction:
    enabled: true                   # Use batch corrected data
```

### Train-Test Split Configuration
```yaml
data:
  train_test_split:
    # Split method - IMPORTANT:
    # - "random": Standard random split (ignores train/test settings below)
    # - "sex": Train on one sex, test on another
    # - "tissue": Train on one tissue, test on another
    method: "random"                # Default: mixed random split
    test_split: 0.2                 # Used only when method="random"
    random_state: 42
    
    # Settings below ONLY used when method="sex" or "tissue"
    train:
      sex: "male"                   # Used only when method="sex"
      tissue: "head"                # Used only when method="tissue"
    test:
      sex: "female"                 # Used only when method="sex"  
      tissue: "body"                # Used only when method="tissue"
```

### Gene Filtering Options
```yaml
gene_preprocessing:
  gene_filtering:
    remove_sex_genes: false         # Remove sex-linked genes
    remove_autosomal_genes: false   # Remove autosomal genes
    highly_variable_genes: false    # Select highly variable genes
    only_keep_lnc_genes: false      # Keep only lncRNA genes
    remove_lnc_genes: false         # Remove lncRNA genes
```

## Usage

### Quick Start
```bash
# Run with default configuration
python run_timeflies.py new

# View available options
python run_timeflies.py new --help

# Run with specific configuration
python run_timeflies.py new --config configs/custom.yaml
```

### Command Line Interface
```bash
# Train a CNN model on head tissue for age prediction
python run_timeflies.py new --tissue head --model cnn --encoding age

# Train with specific gene filtering
python run_timeflies.py new --tissue head --model cnn --encoding age --gene-method hvg

# Cross-sex analysis: train on males, test on females
python run_timeflies.py new --tissue head --model cnn --encoding age --split-method sex

# Use batch corrected data
python run_timeflies.py new --tissue head --model cnn --encoding age --batch-correction
```

### Advanced Configuration
```bash
# Custom cell type and sex filtering
python run_timeflies.py new --tissue head --model cnn --encoding age \
    --cell-type "muscle cell" --sex-type male

# Multiple model comparison
python run_timeflies.py new --tissue head --encoding age --model cnn
python run_timeflies.py new --tissue head --encoding age --model mlp
python run_timeflies.py new --tissue head --encoding age --model logistic
```

## Data Pipeline

The TimeFlies pipeline consists of several automated stages:

1. **Data Loading**: Load h5ad files from `data/raw/h5ad/`
2. **Preprocessing**: Apply filters, normalization, and scaling
3. **Gene Selection**: Filter genes based on configuration
4. **Train-Test Split**: Create training and evaluation sets
5. **Model Training**: Train selected ML model
6. **Evaluation**: Generate predictions and metrics
7. **Visualization**: Create plots and interpretability analysis
8. **Output**: Save results to organized directory structure

### Automatic Path Management
The pipeline automatically:
- Creates organized output directories based on experiment configuration
- Saves processed data to `data/processed/` with consistent naming
- Stores models in `outputs/models/` with metadata
- Generates analysis results in `outputs/results/`

## Supported Models

### Neural Networks
- **CNN (Convolutional Neural Network)**: Captures local gene expression patterns
- **MLP (Multi-Layer Perceptron)**: Standard feedforward neural network

### Traditional ML
- **XGBoost**: Gradient boosting for tabular data
- **Random Forest**: Ensemble method
- **Logistic Regression**: Linear classification

### Model Features
Each model includes:
- Hyperparameter optimization
- Early stopping
- Model checkpointing
- Performance evaluation
- Feature importance analysis

## Batch Correction

### scVI Integration
TimeFlies integrates scVI (Single-cell Variational Inference) for batch correction:
- Removes technical variation while preserving biological signals
- Supports multiple tissues and experimental conditions
- Generates quality metrics and visualizations

### Usage
```bash
# Enable batch correction in configuration
python run_timeflies.py new --batch-correction --tissue head
```

### Quality Assessment
- UMAP visualizations before/after correction
- Batch mixing metrics
- Biological signal preservation analysis

## Analysis and Visualization

### Automated Outputs
The pipeline generates comprehensive analysis including:

### Model Performance
- Classification accuracy and metrics
- Confusion matrices
- ROC curves
- Feature importance rankings (SHAP)

### Biological Insights
- Age-related gene expression changes
- Sex-specific expression patterns
- Cell type-specific aging signatures

### Visualizations
- UMAP and PCA plots
- Training metrics and loss curves
- Gene expression heatmaps
- Statistical summaries

### Interactive Analysis
Use the Jupyter notebook for custom analysis:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Results

All results are automatically organized in the `outputs/` directory:

```
outputs/
├── models/                         # Trained models and metadata
├── results/                        # Analysis results and plots
└── logs/                          # Execution logs
```

### Result Structure
Each experiment creates organized outputs:
- Model files and training history
- Performance metrics and statistics
- Visualization plots (PNG/PDF)
- SHAP interpretation results
- Detailed logs and metadata

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use TimeFlies in your research, please cite:

```bibtex
@software{timeflies2024,
  title={TimeFlies: Machine Learning for Aging Analysis in Drosophila Single-Cell RNA-seq Data},
  author={Singh Lab},
  year={2024},
  url={https://github.com/rsinghlab/TimeFlies}
}
```

## Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Review configuration options in `configs/default.yaml`
- Check the interactive notebook in `notebooks/analysis.ipynb`

## Acknowledgments

- The single-cell genomics community for developing foundational tools
- Contributors to scanpy, scvi-tools, and other open-source packages
- The Drosophila research community for data and biological insights