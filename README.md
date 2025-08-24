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

```bash
# Complete setup workflow
timeflies setup [--batch-correct]

# Train models with evaluation
timeflies train [--with-eda --with-analysis]

# Evaluate trained models
timeflies evaluate [--with-eda --with-analysis]

# Run project-specific analysis
timeflies analyze
```

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

## Key Features

### Machine Learning Pipeline
- **Deep Learning Models**: CNN, MLP architectures for single-cell analysis
- **Traditional ML**: XGBoost, Random Forest, Logistic Regression for comparison studies
- **Automated Evaluation**: Built-in performance metrics and validation
- **Model Interpretability**: Feature importance analysis with SHAP (configurable)

### Data Processing
- **Batch Correction**: scVI-tools integration for technical noise removal
- **Smart Splitting**: Stratified train/eval splits preserving biological structure
- **Quality Control**: Automated data validation and preprocessing

### Research Tools
- **3-Tier Test Data**: Tiny/synthetic/real fixtures for reliable development
- **Comprehensive EDA**: Exploratory data analysis with automated reporting
- **Flexible Configuration**: YAML-based project and model settings

## Commands Reference

**All 11 CLI commands with their full options:**

### Core Research Commands
```bash
timeflies setup [--batch-correct] [--dev]     # Complete setup workflow
timeflies train [--with-eda] [--with-analysis] # Train models  
timeflies evaluate [--with-eda] [--with-analysis] [--interpret] [--visualize] # Evaluate models on test data
timeflies analyze [--predictions-path PATH] [--analysis-script PATH] [--with-eda] # Project-specific analysis scripts
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
timeflies test [unit|integration|functional|system|all] [--coverage] [--fast] [--debug] [--rerun]
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
# Enable/disable SHAP analysis during evaluation
interpretation:
  shap:
    enabled: true          # Enable SHAP interpretation
    n_samples: 100        # Number of samples for SHAP calculation
    feature_names: true   # Include gene names in output
    save_values: true     # Save SHAP values to CSV

# Control visualization generation
visualizations:
  enabled: true           # Enable plot generation
  save_plots: true       # Save plots to files
  plot_formats: ["png", "pdf"]  # Output formats
```

#### Analysis Scripts
Configure project-specific analysis workflows:

```yaml
analysis:
  run_analysis_script:
    enabled: true         # Run project-specific analysis after evaluation
  custom_analysis:        # Custom analysis settings
    save_intermediate: true
    generate_reports: true
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
- `templates/README.md` - Full documentation and examples

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

## Project Structure

```
TimeFlies/
   configs/              # Project configurations
   data/                 # Input datasets (user-provided)
      fruitfly_aging/
      fruitfly_alzheimers/
   outputs/              # Results, models, plots (see Output Structure above)
   src/
      common/           # Shared framework components
      projects/         # Project-specific implementations
   tests/                # Comprehensive test suite
      fixtures/         # Test data (tiny/synthetic/real)
      outputs/          # Test-specific outputs (temporary)
   install_timeflies.sh  # One-click installer
```

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
│   ├── experiments/                   # All experimental results
│   │   ├── uncorrected/              # Original data experiments  
│   │   │   ├── all_runs/             # Complete experiment history
│   │   │   │   └── head_cnn_age/     # Config-specific experiments
│   │   │   │       ├── 2025-01-22_14-30-15/  # Timestamped experiment
│   │   │   │       │   ├── model.h5           # Trained model
│   │   │   │       │   ├── training/          # Training artifacts
│   │   │   │       │   │   ├── history.json   # Training metrics
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
│   │   │   └── best/                 # Symlinks to best experiments
│   │   │       └── head_cnn_age -> ../all_runs/head_cnn_age/2025-01-22_14-30-15/
│   │   └── batch_corrected/          # Batch-corrected data experiments
│   │       ├── all_runs/             # Same structure as uncorrected
│   │       └── best/                 # Best batch-corrected models
│   ├── data/                         # Processed datasets
│   │   ├── splits/                   # Train/eval data splits
│   │   └── batch_corrected/          # scVI-corrected data files
│   ├── analysis/                     # Project-specific analysis
│   │   ├── reports/                  # HTML/PDF reports
│   │   └── custom/                   # User analysis results
│   └── logs/                         # System logs
└── fruitfly_alzheimers/              # Same structure for other projects
    ├── experiments/
    ├── data/
    ├── analysis/
    └── logs/
```

### Key Output Components

**Model Artifacts:**
- `model.h5` - Trained TensorFlow/Keras models ready for deployment
- `training/history.json` - Complete training metrics and loss curves
- `evaluation/metrics.json` - Test performance (accuracy, precision, recall, F1)

**Interpretability Results:**
- `evaluation/shap_values.csv` - Feature importance values for each prediction
- `plots/shap_summary.png` - Gene importance visualization 
- `plots/feature_importance.png` - Top contributing features
- `plots/expression_heatmap.png` - Gene expression patterns

**Model Performance:**
- `plots/confusion_matrix.png` - Classification accuracy breakdown
- `plots/roc_curve.png` - ROC analysis for binary classification
- `evaluation/predictions.csv` - Model predictions with confidence scores

**Best Model Collection:**
The `best/` directory contains symlinks to the highest-performing experiment for each configuration, making it easy to access optimal models without searching through all experiments.

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Run tests: `timeflies test --coverage`
4. Submit pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Singh Lab

Developed by the Singh Lab for advancing aging research through machine learning.

**Contact**: [Singh Lab](https://github.com/rsinghlab)
**Repository**: [TimeFlies](https://github.com/rsinghlab/TimeFlies)

---

*TimeFlies v1.0 - Advancing aging research through machine learning*