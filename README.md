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
- **Batch correction**: Using scVI
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
├── Code/                          # Main source code
│   ├── time_flies.py             # Main entry point
│   ├── pipeline_manager.py       # Pipeline orchestration
│   ├── config.py                 # Configuration settings
│   ├── preprocess.py             # Data preprocessing
│   ├── model.py                  # ML model implementations
│   ├── scvi_batch.py             # scVI batch correction
│   ├── setup.py                  # setup file
│   ├── visuals.py                # Visualization tools
│   ├── interpreter.py            # Model interpretation (SHAP)
│   ├── eda.py                    # Exploratory data analysis
│   └── utilities.py              # Helper functions
├── Data/                          # Data storage
│   ├── h5ad/                     # Single-cell data files
│   │   ├── head/                 # Head tissue data
│   │   ├── body/                 # Body tissue data
│   │   └── all/                  # Combined tissue data
│   ├── gene_lists/               # Gene annotation files
│   └── preprocessed/             # Processed data outputs
├── Models/                        # Trained model storage
├── Analysis/                      # Analysis notebooks and results
│   ├── Analysis.ipynb            # Main analysis notebook
│   ├── UMAP/                     # UMAP visualizations
│   ├── batch_corrected/          # Batch-corrected results
│   └── uncorrected/              # Uncorrected results
├── Requirements/                  # Environment specifications
│   ├── linux/requirements.txt    # Linux dependencies
|   ├── linux/batch_environment.yml # Linux Batch environment
│   ├── macOS/mac_gpu.yml         # macOS environment
│   └── windows/Windows_GPU.yml   # Windows environment
└── README.md                      # This file
```

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

#### 2. Set Up Environments

**Linux:**
```bash
# Main pipeline environment
conda create -n timeflies python=3.12.1
conda activate timeflies
pip install -r Requirements/linux/requirements.txt

# Batch correction environment (separate due to dependency conflicts)
conda env create -f Requirements/linux/batch_environment.yml
```

**macOS:**
```bash
conda env create -f Requirements/macOS/mac_gpu.yml
conda activate timeflies
```

**Windows:**
```bash
conda env create -f Requirements/windows/Windows_GPU.yml
conda activate timeflies
```

#### 3. Verify Installation
```bash
# Test main environment
conda activate timeflies
python -c "import tensorflow, pandas, numpy, scanpy; print('Main environment ready!')"

# Test batch correction environment
conda activate batch_environment
python -c "import scvi, scanpy; print('Batch correction environment ready!')"
```

### Environment Management

**Two separate environments are required:**
- **`timeflies`**: Main pipeline (CNN, MLP, data processing, visualization)
- **`batch_environment`**: scVI batch correction (has conflicting dependencies)

**Switch between environments:**
```bash
conda activate timeflies      # For main pipeline
conda activate batch_environment  # For batch correction only
```

### Key Dependencies Installed

**Main Environment (`timeflies`):**
- `tensorflow`: Neural network models (CNN, MLP)
- `scikit-learn`: Traditional ML models (XGBoost, Random Forest)
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `shap`: Model interpretation
- `scanpy`: Single-cell analysis (shared with batch env)

**Batch Correction Environment (`batch_environment`):**
- `scvi-tools`: Batch correction using variational inference
- `scanpy`: Single-cell analysis
- `pytorch`: Backend for scVI models
- Compatible versions that work with scVI requirements

## Configuration

The main configuration is in `Code/config.py`. Key settings include:

### Basic Settings
```python
"DataParameters": {
    "GeneralSettings": {
        "tissue": "head",                    # 'head', 'body', 'all'
        "model_type": "CNN",                 # 'CNN', 'MLP', 'XGBoost', etc.
        "encoding_variable": "age",          # 'age', 'sex', 'sex_age'
        "cell_type": "all",                  # Cell type filter
        "sex_type": "all",                   # 'all', 'male', 'female'
    },
    "BatchCorrection": {
        "enabled": True,                     # Use batch corrected data
    },
}
```

### Gene Filtering Options
```python
"GenePreprocessing": {
    "GeneFiltering": {
        "remove_sex_genes": False,           # Remove sex-linked genes
        "remove_autosomal_genes": False,     # Remove autosomal genes
        "highly_variable_genes": False,      # Select highly variable genes
        "remove_lnc_genes": False,          # Remove lncRNA genes
    }
}
```

### Model Training
```python
"Training": {
    "epochs": 100,
    "batch_size": 512,
    "learning_rate": 0.001,
    "early_stopping_patience": 10,
}
```

## Usage

### Initial Setup (Required)

**Before running any analysis, you must first run the setup script:**

```bash
cd Code
conda activate timeflies
python setup.py
```

**What setup.py does:**
- **Stratified Splitting**: Creates balanced train/evaluation splits from original h5ad files
- **Data Integrity**: Ensures proper representation across age groups, sex, and other factors
- **File Organization**: Saves split datasets to `Data/hda5/uncorrected/` directory
- **Reproducibility**: Uses configured random seed for consistent splits across runs
- **Validation**: Verifies the integrity and balance of created splits

**Configuration for setup.py:**
The setup process uses settings from `Code/config.py`:
```python
"Setup": {
    "strata": "age",                        # Column used for stratification
    "tissue": "head",                       # Tissue type (e.g., 'head', 'body')
    "seed": 42,                             # Random seed for reproducibility
    "split_size": 5000,                     # Number of samples to split for evaluation
},
```

**Output files created:**
- `Data/hda5/uncorrected/{tissue}/fly_train.h5ad`
- `Data/hda5/uncorrected/{tissue}/fly_eval.h5ad`

### Basic Usage

**Complete Workflow:**

1. **Configure the pipeline** by editing `Code/config.py`
2. **Run initial data setup** (Required first step):
```bash
cd Code
conda activate timeflies
python setup.py
```
3. **Optional: Run batch correction**:
```bash
conda activate batch_environment
python scvi_batch.py --train --tissue head
```
4. **Enable batch correction in config.py** (if used):
```python
"BatchCorrection": {"enabled": True}
```
5. **Run the main pipeline**:
```bash
conda activate timeflies
python time_flies.py
```

### Advanced Usage

**Cross-sex analysis:**
```python
# In config.py, set up cross-training
"TrainTestSplit": {
    "method": "sex",
    "train": {"sex": "male"},
    "test": {"sex": "female"},
}
```

**Custom analysis:**
```python
# Use the Analysis.ipynb notebook for interactive exploration
jupyter notebook Analysis/Analysis.ipynb
```

## Data Pipeline

The pipeline consists of several stages:

1. **Initial Setup** (Required): Run `setup.py` to create stratified train/eval splits
2. **Data Loading**: Load h5ad files containing single-cell data
3. **Batch Correction** (optional): Remove batch effects using scVI - done after setup
4. **Filtering**: Apply cell and gene filters based on configuration
5. **Preprocessing**: Normalization, scaling, highly variable gene selection
6. **Model Training**: Train selected ML model
7. **Evaluation**: Generate predictions and metrics
8. **Visualization**: Create plots and interpretability analysis

## Supported Models

### Neural Networks
- **CNN (Convolutional Neural Network)**: For capturing local gene expression patterns
- **MLP (Multi-Layer Perceptron)**: Standard feedforward neural network

### Traditional ML
- **XGBoost**: Gradient boosting for tabular data
- **Random Forest**: Ensemble method
- **Logistic Regression**: Linear classification

### Model Selection
Models are automatically configured based on the `model_type` setting in config.py. Each model includes:
- Hyperparameter optimization
- Early stopping
- Model checkpointing
- Performance evaluation

## Batch Correction

### Methods Available

#### scVI (Single-cell Variational Inference)
- Deep generative model for batch correction
- Preserves biological variation while removing technical effects
- Configurable latent dimensions and training parameters

### Usage

**Important**: Batch correction is performed during the setup phase, before running the main pipeline.

The batch correction script has three distinct modes:

1. **Train**: Performs batch correction on raw data and saves corrected files
2. **Evaluate**: Computes integration quality metrics on corrected data
3. **Visualize**: Creates UMAP plots comparing before/after correction

**File Selection & Evaluation Strategy**:

- **Default (Recommended)**: Uses train/eval split files created by `setup.py`

- **With `--original`**: Uses full original `{tissue}_data.h5ad` files directly
  - **Complete Data**: Uses all available cells for batch correction
  - **Manuscript/Final Analysis**: Useful for final results using complete datasets
  - **No Holdout**: Cannot evaluate correction quality on unseen data
  - **Higher Resource Usage**: Requires more memory and computational time
  - **Bypasses Setup**: Skips the stratified splitting step entirely


**Important**: Make sure to activate the batch correction environment first:
```bash
conda activate batch_environment
```

```bash
# 1. Train scVI model (uses uncorrected data automatically)
python scvi_batch.py --train --tissue head  # head is default tissue

# 2. Evaluate correction quality (uses batch-corrected data automatically)  
python scvi_batch.py --evaluate --tissue head

# 3. Generate comparison visualizations (loads both data types)
python scvi_batch.py --visualize --tissue head

# Or use other tissues
python scvi_batch.py --train --tissue body
python scvi_batch.py --train --tissue all

# Use full original file instead of train/eval splits
python scvi_batch.py --train --original --tissue head
python scvi_batch.py --evaluate --original --tissue head
python scvi_batch.py --visualize --original --tissue head
```

### Quality Assessment
- UMAP visualizations before/after correction
- Batch mixing metrics
- Biological signal preservation

## Analysis and Visualization

### Exploratory Data Analysis (EDA)
- Cell count distributions
- Gene expression statistics
- Quality control metrics
- Outlier detection

### Dimensionality Reduction
- **PCA**: Principal component analysis
- **UMAP**: Uniform Manifold Approximation and Projection
- Interactive plots colored by age, sex, or cell type

### Model Interpretation
- **SHAP values**: Feature importance for individual predictions
- **Confusion matrices**: Classification performance
- **ROC curves**: Model evaluation metrics

### Generated Outputs
All visualizations are saved to the `Analysis/` directory:
- `UMAP/`: Dimensionality reduction plots
- `batch_corrected/`: Results after batch correction
- `uncorrected/`: Results from original data

## Results

The pipeline generates comprehensive results including:

### Model Performance
- Classification accuracy across age groups
- Cross-sex generalization performance
- Feature importance rankings

### Biological Insights
- Age-related gene expression changes
- Sex-specific expression patterns
- Cell type-specific aging signatures

### Quality Metrics
- Batch correction effectiveness
- Model interpretability scores
- Statistical significance tests

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
- Check the documentation in the `Analysis/` directory
- Review configuration options in `Code/config.py`

## Acknowledgments

- The single-cell genomics community for developing foundational tools
- Contributors to scanpy, scvi-tools, and other open-source packages
- The Drosophila research community for data and biological insights