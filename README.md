# TimeFlies v1.0

**Machine Learning for Aging Analysis in Drosophila Single-Cell Data**

TimeFlies is a comprehensive machine learning framework for analyzing aging patterns in Drosophila single-cell RNA sequencing data. It provides deep learning models, model interpretability analysis, batch correction capabilities, and a complete research workflow.

## Quick Start

### Installation

```bash
# Download and run the installer
curl -O https://raw.githubusercontent.com/rsinghlab/TimeFlies/main/install_timeflies.sh
chmod +x install_timeflies.sh
./install_timeflies.sh
```

### Activate Environment

```bash
source .activate.sh
```

### Basic Usage

**Command Line Interface:**
```bash
# Complete setup workflow
timeflies setup [--batch-correct]

# Train models with evaluation
timeflies train [--with-eda --with-analysis]

# Evaluate trained models
timeflies evaluate [--with-eda --with-analysis]

# Automated multi-model training queue
timeflies queue [configs/model_queue.yaml] [--no-resume]

# Run project-specific analysis
timeflies analyze
```

**Graphical User Interface (Alternative):**
For users who prefer a GUI, run `python TimeFlies_Launcher.py` to open a user-friendly graphical interface with:
- Installation wizard with progress tracking
- Point-and-click analysis workflow
- Configuration editing (project, tissue, model settings)
- Results browser and visualization tools
- Built-in help and documentation

Both interfaces provide identical functionality - choose what works best for you.

## Research Workflow

1. **Data Setup**: Place your `*_original.h5ad` files in `data/[project]/[tissue]/`
2. **Configuration**: Edit `configs/default.yaml` for your project settings
3. **Setup**: Run `timeflies setup` to create train/eval splits and verify system
4. **Training**: Run `timeflies train` for model training with automatic evaluation
5. **Evaluation**: Run `timeflies evaluate` to assess model performance on test data
6. **Analysis**: Results available in `outputs/[project]/` with model interpretability

## Supported Projects

- **Fruitfly Aging**: Healthy aging analysis in Drosophila head tissue
- **Fruitfly Alzheimer's**: Disease model analysis with neurodegeneration patterns
- **Custom Projects**: Any single-cell transcriptomics data in AnnData format

## Key Features

### Machine Learning Pipeline
- **Deep Learning Models**: CNN, MLP architectures for single-cell analysis
- **Traditional ML**: XGBoost, Random Forest, Logistic Regression for comparison studies
- **Automated Evaluation**: Built-in performance metrics and validation
- **Model Interpretability**: Feature importance analysis with SHAP (configurable)
- **Model Queue System**: Automated sequential training of multiple models with different configurations

### Data Processing
- **Batch Correction**: scVI-tools integration for technical noise removal
- **Smart Splitting**: Stratified train/eval splits preserving biological structure
- **Quality Control**: Automated data validation and preprocessing

### Research Tools
- **3-Tier Test Data**: Tiny/synthetic/real fixtures for reliable development
- **Comprehensive EDA**: Exploratory data analysis with automated reporting
- **Flexible Configuration**: YAML-based project and model settings

### Automated Model Queue System
- **Sequential Training**: Train multiple models automatically with progress tracking
- **Configuration Overrides**: Per-model settings for preprocessing, hyperparameters, and analysis options
- **Checkpoint/Resume**: Automatic saving and resuming of interrupted training sessions
- **Comprehensive Reports**: Markdown summaries and CSV exports for model comparison
- **Flexible Preprocessing**: Different batch correction, filtering, and splitting methods per model

## Documentation

### Comprehensive Guides
- **[Model Queue System Guide](docs/model_queue_guide.md)** - Complete guide for automated multi-model training
- **[Analysis Templates Guide](templates/README.md)** - Custom analysis script templates and examples
- **[Development Roadmap](docs/timeflies_todo.md)** - Current development status and future plans

### Quick Links
- **Templates**: Pre-built analysis scripts in `templates/` directory
- **Configuration**: YAML examples in `configs/` directory
- **Test Data**: 3-tier test fixtures in `tests/fixtures/`

## Commands Reference

**All 12 CLI commands with their full options:**

### Core Research Commands
```bash
timeflies setup [--batch-correct] [--dev]     # Complete setup workflow
timeflies train [--with-eda] [--with-analysis] # Train models
timeflies evaluate [--with-eda] [--with-analysis] [--interpret] [--visualize] # Evaluate models on test data
timeflies analyze [--predictions-path PATH] [--analysis-script PATH] [--with-eda] # Project-specific analysis scripts
timeflies queue [configs/model_queue.yaml] [--no-resume] # Automated multi-model training queue (see docs/model_queue_guide.md)
```

### Data & Analysis Commands
```bash
timeflies split                               # Create train/eval splits
timeflies eda [--save-report]                 # Exploratory data analysis
timeflies batch-correct                       # Create batch-corrected files (requires .venv_batch)
timeflies verify                              # System verification
```

### Development Commands
```bash
timeflies test [unit|integration|functional|system|all] [--coverage] [--verbose] [--fast] [--debug] [--rerun]
timeflies create-test-data [--tier tiny|synthetic|real|all] [--cells N] [--genes N] [--batch-versions]
timeflies update                              # Update to latest version
```

### Global Options (work with any command)
```bash
--verbose                 # Detailed logging
--batch-corrected         # Use existing batch-corrected data (any command)
--tissue head|body        # Override tissue type
--model CNN|MLP|xgboost|random_forest|logistic   # Override model type
--target age              # Override target variable
--aging                   # Use fruitfly_aging project
--alzheimers              # Use fruitfly_alzheimers project
```

## Configuration

TimeFlies uses YAML configuration files to control model training, evaluation, and analysis settings. The main configuration is in `configs/default.yaml`.

### Key Configuration Sections

#### Model Interpretability (SHAP Analysis)
Control SHAP interpretation and visualizations:

```yaml
# Feature importance analysis
interpretation:
  shap:
    enabled: false           # Enable/disable SHAP interpretation (includes visualizations)
    load_existing: false     # Load existing SHAP values instead of computing
    reference_size: 100      # Reference size for SHAP analysis

# Visualizations
visualizations:
  enabled: true             # Enable general visualizations (training plots, confusion matrix, ROC curves, etc.)
```

#### Analysis Scripts
Configure project-specific analysis workflows:

```yaml
analysis:
  # Exploratory data analysis
  eda:
    enabled: false

  # Run project-specific analysis scripts
  run_analysis_script:
    enabled: false  # Set to true to run project-specific analysis after training
```

### CLI Overrides
Override configuration settings using command-line flags:

```bash
# Force SHAP interpretation (overrides config)
timeflies evaluate --interpret

# Force visualizations (overrides config)
timeflies evaluate --visualize

# Use custom analysis script
timeflies analyze --analysis-script templates/my_custom_analysis.py

# Combine flags
timeflies evaluate --interpret --visualize --with-analysis
```

### Custom Analysis Scripts
Create custom analysis workflows using templates:

```bash
# Copy template and customize
cp templates/aging_analysis_template.py templates/my_analysis.py

# Run your custom analysis
timeflies analyze --analysis-script templates/my_analysis.py
```

Available templates:
- `templates/custom_analysis_example.py` - Basic template with all features
- `templates/aging_analysis_template.py` - Aging-specific analysis patterns
- **[templates/README.md](templates/README.md)** - Full documentation and examples

## Development

### For Developers

```bash
# Clone repository
git clone https://github.com/rsinghlab/TimeFlies.git
cd TimeFlies

# Setup development environments (creates .venv + .venv_batch with all dependencies)
python3 run_timeflies.py setup --dev

# Activate development environment
source .activate.sh

# Now you can use timeflies command directly
timeflies verify
timeflies test --coverage
timeflies create-test-data --tier tiny  # (optional - already included)

# For batch correction development (specialized)
source .activate_batch.sh  # PyTorch + scVI environment (separate from main)
timeflies batch-correct    # Creates batch-corrected versions of test data
```

### Test Data System
- **Tiny**: 50 cells, 100 genes (committed, fast CI/CD)
- **Synthetic**: 500 cells, 1000 genes (generated from metadata)
- **Real**: 5000 cells, 2000 genes (performance testing)

```bash
timeflies create-test-data --tier tiny --batch-versions     # (optional - already committed)
timeflies create-test-data --tier synthetic --batch-versions # Generate on-demand for testing
```

## Repository Structure

```
TimeFlies/
├── configs/              # YAML configuration files
├── src/                  # Source code
│   └── common/          # Framework components
│       ├── analysis/    # EDA and visualization tools
│       ├── cli/         # Command-line interface
│       ├── core/        # Pipeline and configuration management
│       ├── data/        # Data loading and preprocessing
│       ├── evaluation/  # Model evaluation and metrics
│       ├── models/      # ML model implementations
│       └── utils/       # Utilities and helpers
├── tests/               # Test suite with 3-tier test data
│   ├── fixtures/        # Test data (tiny/synthetic/real)
│   └── outputs/         # Test outputs (temporary)
├── templates/           # Analysis script templates
├── docs/               # Documentation and notebooks
├── install_timeflies.sh # One-click installer
├── run_timeflies.py    # Main CLI entry point
└── TimeFlies_Launcher.py    # GUI Launcher

```

## User Setup Guide

After installation, users work with this structure in their project directory:

```
your_project/
├── config.yaml          # Created by TimeFlies setup (customize your settings)
├── templates/           # Analysis script templates (created by setup)
│   ├── aging_analysis_template.py
│   ├── custom_analysis_example.py
│   └── README.md
├── data/                # Your input datasets
│   ├── fruitfly_aging/
│   │   └── head/
│   │       ├── *_original.h5ad     # Your raw data files
│   │       ├── *_train.h5ad        # Generated by 'split' command
│   │       └── *_eval.h5ad         # Generated by 'split' command
│   └── fruitfly_alzheimers/
│       └── head/
│           └── *_original.h5ad     # Your raw data files
└── outputs/             # All results generated by TimeFlies
    └── [see Output Structure below]
```

### Getting Started
1. **Install TimeFlies**: `curl -O https://raw.githubusercontent.com/.../install_timeflies.sh && chmod +x install_timeflies.sh && ./install_timeflies.sh`
2. **Activate**: `source .activate.sh` (installs timeflies command to system)
3. **Add your data**: Place `*_original.h5ad` files in `data/[project]/[tissue]/`
4. **Setup**: `timeflies setup` (creates config.yaml, templates/, splits data, verifies system)
5. **Configure**: Edit `config.yaml` for your project settings (or use GUI: `python TimeFlies_Launcher.py`)
6. **Run workflow**: `timeflies train && timeflies evaluate`

## System Requirements

- **Python**: 3.12+
- **OS**: Linux, macOS, Windows (WSL2)
- **Memory**: 8GB+ recommended for larger datasets
- **Storage**: 2GB+ for environments and test data

## Research Applications

TimeFlies is designed for researchers studying:
- Aging mechanisms in model organisms
- Single-cell transcriptomic changes over time
- Disease models and neurodegeneration
- Cross-tissue aging comparisons
- Batch effect correction in sc-RNA-seq

## Output Structure

TimeFlies organizes all outputs in a structured hierarchy for easy navigation and analysis:

```
outputs/
├── fruitfly_aging/                    # Project-specific outputs
│   ├── experiments/                   # Machine learning experiments
│   │   ├── uncorrected/              # Original data experiments
│   │   │   ├── all_runs/             # Complete experiment history
│   │   │   │   └── head_cnn_age/     # Config-specific experiments (tissue_model_target)
│   │   │   │       ├── 2025-01-22_14-30-15/  # Timestamped experiment
│   │   │   │       │   ├── model.h5           # Trained model
│   │   │   │       │   ├── training/          # Training artifacts
│   │   │   │       │   │   ├── history.json   # Training metrics & loss curves
│   │   │   │       │   │   ├── logs/          # Training logs
│   │   │   │       │   │   └── plots/         # Training visualizations
│   │   │   │       │   ├── evaluation/        # Test results
│   │   │   │       │   │   ├── metrics.json   # Performance metrics
│   │   │   │       │   │   ├── predictions.csv # Model predictions
│   │   │   │       │   │   ├── shap_values.csv # SHAP interpretability
│   │   │   │       │   │   └── plots/         # Result visualizations
│   │   │   │       │   │       ├── confusion_matrix.png
│   │   │   │       │   │       ├── roc_curve.png
│   │   │   │       │   │       ├── shap_summary.png
│   │   │   │       │   │       ├── feature_importance.png
│   │   │   │       │   │       └── expression_heatmap.png
│   │   │   │       │   └── metadata.json      # Experiment configuration
│   │   │   │       └── 2025-01-22_15-45-30/  # Another experiment
│   │   │   ├── latest -> all_runs/head_cnn_age/2025-01-22_15-45-30/  # ⚡ Latest experiment
│   │   │   └── best/                 # 🏆 Collection of symlinks to best experiments
│   │   │       └── head_cnn_age -> ../all_runs/head_cnn_age/2025-01-22_14-30-15/
│   │   └── batch_corrected/          # Batch-corrected data experiments (same structure)
│   │       ├── all_runs/
│   │       ├── latest -> all_runs/.../
│   │       └── best/
│   ├── eda/                          # 📊 Exploratory Data Analysis
│   │   ├── uncorrected/
│   │   │   └── head/                 # Tissue-specific EDA
│   │   │       ├── eda_report.html   # Interactive HTML report
│   │   │       ├── eda_summary.json  # Statistical summaries
│   │   │       ├── age_distribution.png
│   │   │       ├── correlation_matrix.png
│   │   │       ├── dimensionality_reduction.png
│   │   │       └── top_expressed_genes.png
│   │   └── batch_corrected/          # EDA for batch-corrected data
│   │       └── head/
│   ├── analysis/                     # Project-specific analysis
│   │   ├── reports/                  # Custom analysis HTML/PDF reports
│   │   └── custom/                   # User analysis script results
│   ├── model_queue_summaries/        # 🚀 Automated model queue results
│   │   └── 2025-01-22_16-30-45/     # Timestamped queue run
│   │       ├── summary_report.md     # Comparative analysis report
│   │       ├── metrics_summary.csv   # Performance metrics for all models
│   │       ├── queue_checkpoint.json # Resume checkpoint data
│   │       └── queue_config.yaml     # Configuration used for this run
│   └── logs/                         # System logs
└── fruitfly_alzheimers/              # Same structure for other projects
    ├── experiments/
    ├── eda/
    ├── analysis/
    ├── model_queue_summaries/
    └── logs/
```

### Key Output Components

**Model Artifacts:**
- `model.h5` - Trained TensorFlow/Keras models ready for deployment
- `training/history.json` - Complete training metrics and loss curves
- `evaluation/metrics.json` - Test performance (accuracy, precision, recall, F1)
- `metadata.json` - Experiment configuration and reproducibility info

**Interpretability Results:**
- `evaluation/shap_values.csv` - Feature importance values for each prediction
- `plots/shap_summary.png` - Gene importance visualization
- `plots/feature_importance.png` - Top contributing features
- `plots/expression_heatmap.png` - Gene expression patterns

**Model Performance:**
- `plots/confusion_matrix.png` - Classification accuracy breakdown
- `plots/roc_curve.png` - ROC analysis for binary classification
- `evaluation/predictions.csv` - Model predictions with confidence scores

**Exploratory Data Analysis (EDA):**
- `eda/*/eda_report.html` - Interactive HTML reports with comprehensive data analysis
- `age_distribution.png`, `correlation_matrix.png` - Data distribution visualizations
- `dimensionality_reduction.png` - t-SNE/UMAP plots
- `eda_summary.json` - Statistical summaries and data quality metrics

**Model Queue Summaries (Automated Multi-Model Training):**
- `model_queue_summaries/*/summary_report.md` - Comparative analysis of all models in queue
- `metrics_summary.csv` - Performance metrics table (accuracy, precision, recall, F1) for all models
- `queue_checkpoint.json` - Resume checkpoint for interrupted queue runs
- `queue_config.yaml` - Configuration file used for the specific queue run

**Smart Navigation:**
- `latest/` - Symlink pointing to most recent experiment for each configuration
- `best/` - Collection of symlinks to highest-performing experiments by configuration
- Both symlinks automatically update as new experiments are run, providing quick access without searching

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `timeflies test --coverage`
4. Submit pull request

## License

This project is licensed under the **TimeFlies Academic Research License** with pre-publication restrictions - see the [LICENSE](LICENSE) file for details.

**Pre-Publication Period:** All rights reserved. Commercial use, redistribution, and derivative works require explicit written permission from the Singh Lab, Brown University.

**Post-Publication:** License will transition to a more permissive open-source license after publication of associated research.

## Singh Lab

Developed by the Singh Lab for advancing aging research through machine learning.

**Contact**: [Singh Lab](https://github.com/rsinghlab)
**Repository**: [TimeFlies](https://github.com/rsinghlab/TimeFlies)

---

*TimeFlies v1.0 - Advancing aging research through machine learning*
