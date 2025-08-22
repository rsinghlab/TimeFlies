# TimeFlies 🧬

**Machine Learning for Aging Analysis**

A modern machine learning framework for analyzing aging patterns in Drosophila single-cell RNA sequencing data using deep learning with comprehensive visualization and SHAP interpretability.

## 🚀 Quick Start

### 1. Install TimeFlies

**Option A: One-Click Installation (Recommended)**
```bash
# Download and run installer
curl -O https://raw.githubusercontent.com/your-username/TimeFlies/main/install_timeflies.sh
bash install_timeflies.sh
source activate.sh
```

**Option B: GUI Installation (User-Friendly)**
1. Download the installer script from GitHub
2. Double-click to run in Terminal (macOS/Linux) 
3. Follow the interactive prompts
4. Launch from Desktop shortcut (auto-created)

### 2. Add Your Data
```bash
# Add your H5AD data files to the appropriate directories
mkdir -p data/fruitfly_aging/head  # or fruitfly_alzheimers
# Copy your *_original.h5ad files into data/[project]/[tissue]/
```

### 3. Run Complete Analysis
```bash
# Verify setup and create data splits
timeflies verify
timeflies split

# Train models with automatic evaluation and SHAP analysis
timeflies train

# Or run everything in one command
timeflies setup-all  # Recommended for first-time users
```

## 📁 Project Structure

After installation and first run, TimeFlies creates this structure:

```
my_aging_research/
├── configs/
│   └── default.yaml          # Main configuration file
├── data/                     # Your input data
│   ├── fruitfly_aging/head/
│   │   └── your_data_original.h5ad
│   └── fruitfly_alzheimers/head/
│       └── your_data_original.h5ad
└── outputs/                  # All results (auto-created)
    ├── fruitfly_aging/
    │   ├── models/          # Trained models
    │   ├── results/         # SHAP analysis & plots
    │   └── logs/            # Training logs
    └── fruitfly_alzheimers/
        └── [similar structure]
```

## 🎯 Key Features

- **🔄 One-Command Setup**: `timeflies setup-all` handles everything  
- **🧬 Smart Data Handling**: Auto-detects projects and creates stratified splits
- **🔒 Safe Operations**: Never overwrites existing data
- **📊 Comprehensive Analysis**: Training + evaluation + SHAP interpretation  
- **🎨 Rich Visualizations**: Automated plots and analysis reports
- **🧪 Robust Testing**: Built-in test suite and verification

## 📋 Core Commands

| Command | Description |
|---------|-------------|
| `verify` | Verify environment and data setup |
| `split` | Create stratified train/eval splits |
| `train` | Train deep learning models with auto-evaluation |
| `evaluate` | Generate SHAP analysis and visualizations |
| `analyze` | Run project-specific analysis |
| `setup-all` | Complete setup: verify + splits + directories |
| `test` | Run comprehensive test suite |

> 📖 **Detailed CLI reference**: All commands support `--help` for full options

## 🔧 Configuration

### Switch Projects
Edit `configs/default.yaml`:
```yaml
project: fruitfly_aging  # or fruitfly_alzheimers
```

### Customize Analysis
```yaml
data:
  tissue: head
  model_type: CNN
  encoding_variable: age

training:
  epochs: 100
  batch_size: 32
```

## 🧬 Supported Analysis Types

### Multi-Project Support
- **Fruitfly Aging**: Age-related gene expression analysis (5, 30, 50, 70 days)
- **Fruitfly Alzheimers**: Disease condition classification
- **Extensible**: Easy to add custom projects

### Advanced Features
- **Multi-level Stratification**: Disease/age + sex + cell type
- **Batch Correction Support**: Handles both original and corrected data
- **SHAP Interpretability**: Model decision analysis with visualizations
- **Comprehensive Testing**: Unit, integration, and end-to-end tests

## 🧪 Development & Testing

```bash
# Run all tests
timeflies test

# Run specific test types  
timeflies test --type unit
timeflies test --fast
```

## 📊 Example Workflow

```bash
# 1. Install and setup
bash install_timeflies.sh
source activate.sh

# 2. Add data
mkdir -p data/fruitfly_aging/head
cp /path/to/your_data_original.h5ad data/fruitfly_aging/head/

# 3. Complete analysis
timeflies setup-all
timeflies train

# 4. View results
ls outputs/fruitfly_aging/results/  # SHAP plots, metrics, analysis
```

## 📓 Jupyter Analysis

The installation includes a comprehensive Jupyter notebook for interactive analysis:

```bash
# Install Jupyter (if needed)
pip install jupyter

# Launch analysis notebook
jupyter notebook docs/notebooks/analysis.ipynb
```

The notebook provides:
- Data exploration and quality control
- Dimensionality reduction (PCA, UMAP) 
- Gene expression analysis across age/sex
- Model performance evaluation
- SHAP feature importance visualization

## ⚙️ Requirements

- **Python**: 3.12+ required
- **Hardware**: GPU support optional, CPU fallback available  
- **Memory**: 8GB+ RAM recommended for large datasets
- **Storage**: ~2GB for dependencies, varies by dataset size

## 🔬 Research Applications

TimeFlies is designed for:
- **Aging Research**: Temporal gene expression analysis
- **Disease Studies**: Classification and biomarker discovery
- **Comparative Analysis**: Cross-tissue and cross-condition studies
- **Method Development**: Extensible framework for new approaches

## 🏫 Research Lab Support

This framework was developed for aging research applications. Contact your lab administrator for:
- Data access and sharing protocols
- Computational resource allocation
- Research collaboration opportunities

## 📄 License

See `LICENSE` file for details.

## 🚀 Future Development

- PyTorch integration for improved GPU utilization
- Additional model architectures (transformers, graph neural networks)
- Real-time analysis dashboard
- Cloud deployment options

---

**Ready to analyze aging patterns in your single-cell data? Get started with `bash install_timeflies.sh`!**