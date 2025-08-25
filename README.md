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

### Automatic Environment Activation

The installer automatically activates TimeFlies. For new terminal windows:

```bash
source .activate.sh
```

### Basic Usage

**Command Line Interface:**
```bash
# Complete setup workflow (customize configs/setup.yaml first)
timeflies setup [--batch-correct]

# Batch correction (automatic environment switching)
timeflies batch-correct

# Train models with automatic evaluation
timeflies train [--with-eda --with-analysis --batch-corrected]

# Evaluate trained models
timeflies evaluate [--with-eda --with-analysis]

# Automated multi-model training queue
timeflies queue [configs/model_queue.yaml] [--no-resume]

# Automated hyperparameter tuning (customize configs/hyperparameter_tuning.yaml first)
timeflies tune [--no-resume]

# Run project-specific analysis
timeflies analyze
```

**Graphical User Interface (Alternative):**
For users who prefer a GUI, run `python TimeFlies_Launcher.py` to open a user-friendly graphical interface with:
- Installation wizard with progress tracking
- Point-and-click analysis workflow
- Batch correction tab with automatic environment switching
- Configuration editing (project, tissue, model settings)
- Hyperparameter tuning interface
- Results browser and visualization tools
- Built-in help and documentation

**GUI Requirements:** The GUI requires tkinter (automatically checked during `timeflies update`).
If missing, install with: `sudo apt install python3-tk` (Ubuntu/Debian) or equivalent for your OS.

Both CLI and GUI interfaces provide identical functionality - choose what works best for you.

## Configuration Files

TimeFlies uses modular configuration files in the `configs/` directory:
- **default.yaml**: Main configuration (project, model, data settings)
- **setup.yaml**: Data splitting settings (split_size, stratify_by)
- **batch_correction.yaml**: Batch correction and PyTorch settings
- **hyperparameter_tuning.yaml**: Hyperparameter search ranges for tuning
- **model_queue.yaml**: Sequential model training configurations

## Research Workflow

1. **Data Setup**: Place your `*_original.h5ad` files in `data/[project]/[tissue]/`
2. **Configuration**: Edit `configs/setup.yaml` for data splitting (split_size, stratify_by, etc.)
3. **Setup**: Run `timeflies setup` to create train/eval splits and verify system
4. **Training**: Run `timeflies train` for model training with automatic evaluation
5. **Evaluation**: Run `timeflies evaluate` to assess model performance on test data
6. **Analysis**: Results available in `outputs/[project]/` with model interpretability

### Re-splitting Data

If you need to change splitting parameters (e.g., different stratification or split size):

```bash
# Edit configs/setup.yaml with new parameters
timeflies setup --force-split    # Recreates splits, preserves batch-corrected files
```

**Smart behavior:**
- `--force-split` removes existing train/eval splits and recreates them
- Batch-corrected files are preserved (never deleted)
- Verifies batch-corrected files still match new splits exactly

## Output Structure

TimeFlies generates comprehensive outputs organized by project and analysis type:

```
outputs/
├── fruitfly_aging/                          # Project-specific results
│   ├── experiments/                         # Model training results
│   │   ├── uncorrected/                     # Non-batch-corrected results
│   │   │   └── all_runs/
│   │   │       └── head_cnn_age/            # Config-specific experiments (tissue_model_target)
│   │   │           ├── 2024-08-25_10-30-15/ # Individual experiment
│   │   │           │   ├── model.h5         # Trained TensorFlow model
│   │   │           │   ├── training/        # Training artifacts
│   │   │           │   │   ├── history.json # Training metrics & loss curves
│   │   │           │   │   ├── logs/        # Training logs
│   │   │           │   │   └── plots/       # Training visualizations
│   │   │           │   ├── evaluation/      # Test results
│   │   │           │   │   ├── metrics.json # Performance metrics
│   │   │           │   │   ├── predictions.csv # Model predictions
│   │   │           │   │   └── plots/       # Performance visualizations
│   │   │           │   │       ├── confusion_matrix.png
│   │   │           │   │       ├── roc_curve.png
│   │   │           │   │       └── classification_report.png
│   │   │           │   ├── shap_analysis/   # SHAP interpretability
│   │   │           │   │   ├── shap_values.csv
│   │   │           │   │   ├── shap_summary.png
│   │   │           │   │   └── feature_importance.png
│   │   │           │   └── metadata.json    # Experiment reproducibility info
│   │   │           ├── latest -> 2024-08-25_10-30-15/  # Symlink to most recent
│   │   │           └── best -> 2024-08-25_10-30-15/    # Symlink to best performance
│   │   ├── batch_corrected/                 # Batch-corrected results (same structure)
│   │   └── queue_experiment_2024-08-25/     # Model queue results
│   │       ├── model_comparison_report.md   # Queue summary report
│   │       ├── model_metrics.csv            # All models comparison
│   │       └── individual_model_results/    # Links to experiment dirs
│   ├── hyperparameter_tuning/               # Hyperparameter optimization
│   │   └── hyperparameter_search_2024-08-25_16-30-45/
│   │       ├── hyperparameter_search_report.md  # Best trials & selection reasoning
│   │       ├── hyperparameter_search_metrics.csv # All trials data for analysis
│   │       ├── checkpoint.json              # Resume capability for interrupted searches
│   │       ├── search_config.yaml           # Configuration backup for reproducibility
│   │       └── optuna_study.db              # Bayesian optimization database (if using Optuna)
│   └── eda/                                 # Exploratory data analysis
│       └── head/                           # Tissue-specific analysis
│           ├── uncorrected/                # Raw data EDA
│           │   ├── eda_report.html         # Interactive analysis report
│           │   ├── plots/                  # EDA visualizations
│           │   │   ├── age_distribution.png
│           │   │   ├── correlation_matrix.png
│           │   │   └── dimensionality_reduction.png
│           │   └── eda_summary.json        # Statistical summaries
│           └── batch_corrected/            # Batch-corrected EDA (same structure)
└── fruitfly_alzheimers/                     # Separate project outputs
    └── [same structure as above]
```

### Key Output Files

- **Experiment Results**: Each training run gets its own timestamped directory with model files, predictions, and analysis
- **Hyperparameter Reports**: Comprehensive analysis of why best parameters were selected with trial comparisons
- **Model Queue Reports**: Comparison across multiple model configurations with links to individual experiments
- **EDA Reports**: Data quality and distribution analysis organized by tissue and batch correction
- **SHAP Analysis**: Model interpretability and feature importance stored within each experiment

## Supported Projects

- **Fruitfly Aging**: Healthy aging analysis in Drosophila head tissue
- **Fruitfly Alzheimer's**: Disease model analysis with neurodegeneration patterns
- **Custom Projects**: Any single-cell transcriptomics data in AnnData format

## Key Features

### Machine Learning Pipeline
- **Deep Learning Models**: CNN, MLP architectures for single-cell analysis
- **Traditional ML**: XGBoost, Random Forest, Logistic Regression for comparison studies
- **Automated Evaluation**: Built-in performance metrics and automatic post-training evaluation
- **Model Interpretability**: Feature importance analysis with SHAP (configurable)
- **Model Queue System**: Automated sequential training of multiple models with different configurations
- **Hyperparameter Tuning**: Grid, random, and Bayesian optimization with CNN architecture variants

### Data Processing
- **Batch Correction**: scVI-tools integration with automatic environment management
  - Per-project enable/disable configuration
  - Proper ML workflow preventing data leakage (train/eval splits)
  - Seamless environment switching for dependencies
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
- **[Hyperparameter Tuning Guide](docs/hyperparameter_tuning_guide.md)** - Grid, random, and Bayesian optimization with CNN variants
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
timeflies setup [--batch-correct] [--force-split] [--dev]     # Complete setup workflow
timeflies train [--with-eda] [--with-analysis] # Train models (includes automatic evaluation)
timeflies evaluate [--with-eda] [--with-analysis] [--interpret] [--visualize] # Evaluate models on test data
timeflies analyze [--predictions-path PATH] [--analysis-script PATH] [--with-eda] # Project-specific analysis scripts
timeflies queue [configs/model_queue.yaml] [--no-resume] # Automated multi-model training queue (see docs/model_queue_guide.md)
timeflies tune [--no-resume] # Hyperparameter optimization using configs/hyperparameter_tuning.yaml (see docs/hyperparameter_tuning_guide.md)
```

### Data & Analysis Commands
```bash
timeflies split [--force-split]              # Create train/eval splits
timeflies eda [--save-report]                 # Exploratory data analysis
timeflies batch-correct                       # Create batch-corrected files (requires .venv_batch)
timeflies verify                              # System verification
```

### Development Commands
```bash
timeflies test [unit|integration|functional|system|all] [--coverage] [--verbose] [--fast] [--debug] [--rerun]
timeflies create-test-data [--tier tiny|synthetic|real|all] [--cells N] [--genes N] [--batch-versions]
```

### System Updates

**Keep TimeFlies Updated**: Use `timeflies update` to get the latest features and bug fixes:

```bash
# Update to latest version from GitHub main branch
timeflies update
```

**What happens during update:**
- Downloads latest TimeFlies code from GitHub
- Updates the installed package via pip
- **Smart file management** - updates system files while preserving your work:

**Files that get UPDATED:**
- `.timeflies_src/` - source code and templates (completely refreshed)
- `TimeFlies_Launcher.py` - GUI launcher (only if content changed)
- Official templates - `README.md`, analysis examples (updated for new features)
- Missing config files - adds new configs like `setup.yaml`, `hyperparameter_tuning.yaml`

**Files that are PRESERVED (never touched):**
- `data/` - your datasets and H5AD files
- `outputs/` - all experiments, analysis results, and trained models
- `configs/` - your customized configuration settings
- Custom templates - any analysis scripts you created

- Requires Git to be installed on your system

**GUI Users**: Use the "Update TimeFlies" button in the Results tab for the same functionality.

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
source .activate_batch.sh  # PyTorch + scVI environment for testing batch correction code
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
├── configs/             # Configuration directory created by TimeFlies setup
│   ├── default.yaml     # Main configuration (customize your settings)
│   ├── setup.yaml       # Data splitting configuration
│   └── ...              # Other config files
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
4. **Setup**: `timeflies setup` (creates configs/, templates/, splits data, verifies system)
5. **Configure**: Edit configs for your project settings (or use GUI: `python TimeFlies_Launcher.py`)
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
