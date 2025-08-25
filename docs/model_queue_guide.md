# Model Queue System - Comprehensive Guide

The TimeFlies Model Queue system enables automated sequential training of multiple models with different configurations, hyperparameters, and preprocessing options. This is essential for comparative analysis and hyperparameter exploration in research.

## Quick Start

1. **Create a queue configuration**: Copy and modify an example configuration
2. **Run the queue**: `timeflies queue configs/your_queue.yaml`
3. **Monitor progress**: See real-time updates and completion status
4. **Review results**: Check `outputs/model_queue_summaries/` for reports

## Configuration Structure

### Basic Configuration Format

```yaml
# Queue settings
queue_settings:
  name: "experiment_name"          # Experiment name for reports
  sequential: true                 # Run models sequentially (true) or parallel (false)
  save_checkpoints: true          # Enable checkpoint/resume functionality
  generate_summary: true          # Generate comparison report at end

# Define models to train
model_queue:
  - name: "model_1"               # Unique model name
    model_type: "CNN"             # Model type: CNN, MLP, xgboost, random_forest, logistic
    description: "Description"     # Optional description
    hyperparameters:              # Model-specific hyperparameters
      epochs: 100
      batch_size: 32
    config_overrides:             # Optional: override global settings
      data:
        batch_correction:
          enabled: true

# Global settings applied to all models (can be overridden)
global_settings:
  project: "fruitfly_aging"       # Project name
  data:                          # Data configuration
    tissue: "head"
    target_variable: "age"
    # ... other data settings
  with_analysis: true            # Run evaluation after training
  interpret: true                # Generate SHAP analysis
  visualize: true               # Create visualizations
```

### Available Configuration Options

#### Model Types
- **`CNN`**: Convolutional Neural Network
- **`MLP`**: Multi-Layer Perceptron
- **`xgboost`**: XGBoost classifier
- **`random_forest`**: Random Forest classifier
- **`logistic`**: Logistic Regression

#### Data Configuration Options (from actual TimeFlies config)
```yaml
data:
  # Core data settings
  model: "CNN"                    # Default model type
  tissue: "head"                  # Tissue type: "head", "body", "all"
  species: "drosophila"          # Species identifier
  cell_type: "all"               # Cell type filter
  sex: "all"                     # Sex filter: "all", "male", "female"
  target_variable: "age"         # Target variable for prediction

  # Batch correction
  batch_correction:
    enabled: false               # Enable batch correction

  # Data filtering
  filtering:
    include_mixed_sex: false     # Include mixed sex samples

  # Data sampling
  sampling:
    samples: null                # Number of samples (null for all)
    variables: null              # Number of genes (null for all)

  # Train/test splitting
  split:
    method: "column"             # Split method: "column" or "random"
    column: "genotype"           # Column to split on (if method="column")
    train: ["control"]           # Values for training
    test: ["ab42", "htau"]       # Values for testing
```

#### Model Training Settings
```yaml
model:
  training:
    epochs: 100                  # Number of training epochs
    batch_size: 32               # Batch size
    validation_split: 0.2        # Validation split ratio
    early_stopping_patience: 8   # Early stopping patience
    learning_rate: 0.001         # Learning rate
```

#### Hyperparameters by Model Type

**CNN/MLP Models:**
```yaml
hyperparameters:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  # CNN specific:
  filters: [32, 64]             # Convolutional filters
  # MLP specific:
  hidden_units: [128, 64]       # Hidden layer sizes
```

**XGBoost Models:**
```yaml
hyperparameters:
  n_estimators: 100             # Number of trees
  max_depth: 6                  # Maximum tree depth
  learning_rate: 0.3            # Learning rate
  subsample: 0.8               # Subsample ratio
  colsample_bytree: 0.8        # Feature subsample ratio
```

**Random Forest Models:**
```yaml
hyperparameters:
  n_estimators: 100             # Number of trees
  max_depth: null               # Maximum depth (null for unlimited)
  min_samples_split: 2          # Minimum samples to split
  min_samples_leaf: 1           # Minimum samples in leaf
```

**Logistic Regression Models:**
```yaml
hyperparameters:
  max_iter: 1000                # Maximum iterations
  C: 1.0                        # Regularization strength
  penalty: "l2"                 # Regularization type
```

## Per-Model Configuration Overrides

You can override any global setting for specific models using `config_overrides`:

```yaml
model_queue:
  - name: "specialized_model"
    model_type: "CNN"
    hyperparameters:
      epochs: 50
    config_overrides:
      # Override project for this model only
      project: "fruitfly_alzheimers"

      # Override data settings
      data:
        sex: "male"              # Only male samples
        batch_correction:
          enabled: true          # Enable batch correction
        split:
          method: "random"       # Use random split
        sampling:
          samples: 5000          # Limit to 5000 samples

      # Override analysis settings
      interpret: false           # Skip SHAP for this model
      with_analysis: false       # Skip evaluation
```

## Example Configurations

### 1. Basic Comparison Queue
```yaml
queue_settings:
  name: "basic_model_comparison"
  sequential: true
  save_checkpoints: true
  generate_summary: true

model_queue:
  - name: "cnn_baseline"
    model_type: "CNN"
    description: "Baseline CNN model"
    hyperparameters:
      epochs: 50
      batch_size: 32

  - name: "xgboost_baseline"
    model_type: "xgboost"
    description: "Baseline XGBoost model"
    hyperparameters:
      n_estimators: 100
      max_depth: 6

  - name: "random_forest_baseline"
    model_type: "random_forest"
    description: "Baseline Random Forest model"
    hyperparameters:
      n_estimators: 100

global_settings:
  project: "fruitfly_aging"
  data:
    tissue: "head"
    target_variable: "age"
  with_analysis: true
  interpret: true
```

### 2. Preprocessing Comparison Queue
```yaml
queue_settings:
  name: "preprocessing_comparison"
  sequential: true
  save_checkpoints: true
  generate_summary: true

model_queue:
  # Default preprocessing
  - name: "cnn_default"
    model_type: "CNN"
    description: "CNN with default preprocessing"
    hyperparameters:
      epochs: 50

  # Batch correction enabled
  - name: "cnn_batch_corrected"
    model_type: "CNN"
    description: "CNN with batch correction"
    hyperparameters:
      epochs: 50
    config_overrides:
      data:
        batch_correction:
          enabled: true

  # Male samples only
  - name: "cnn_male_only"
    model_type: "CNN"
    description: "CNN trained on male samples"
    hyperparameters:
      epochs: 50
    config_overrides:
      data:
        sex: "male"

  # Different splitting strategy
  - name: "cnn_random_split"
    model_type: "CNN"
    description: "CNN with random data splitting"
    hyperparameters:
      epochs: 50
    config_overrides:
      data:
        split:
          method: "random"

global_settings:
  project: "fruitfly_aging"
  data:
    tissue: "head"
    target_variable: "age"
    batch_correction:
      enabled: false
    sex: "all"
    split:
      method: "column"
      column: "genotype"
      train: ["control"]
      test: ["ab42", "htau"]
  with_analysis: true
```

## Usage

### Basic Usage
```bash
# Run a model queue
timeflies queue configs/my_queue.yaml

# Start fresh (ignore any existing checkpoint)
timeflies queue configs/my_queue.yaml --no-resume
```

### Queue Execution Flow
1. **Validation**: Configuration file is validated
2. **Checkpoint Check**: Looks for existing checkpoint to resume
3. **Sequential Training**: Each model is trained in order
4. **Progress Updates**: Shows "X completed, Y remaining"
5. **Checkpoint Saving**: Progress saved after each model
6. **Evaluation**: Each model evaluated if configured
7. **Summary Generation**: Comparison report created at end

### Progress Tracking
During execution, you'll see:
```
============================================================
STARTING MODEL QUEUE: 5 models to train
Queue name: preprocessing_comparison
============================================================

[1/5] Training: cnn_default
Model Type: CNN
Description: CNN with default preprocessing
============================================================

[INFO] Starting training for cnn_default...
[INFO] Starting evaluation for cnn_default...
[OK] Model cnn_default completed in 45.3s

[PROGRESS] 1 completed, 4 remaining

----------------------------------------
Progress: 1 completed, 0 failed
Best model so far: cnn_default (accuracy: 0.847)
----------------------------------------
```

### Checkpoint and Resume
If training is interrupted, you can resume:
```bash
# This will automatically resume from the last completed model
timeflies queue configs/my_queue.yaml

# To start completely fresh
timeflies queue configs/my_queue.yaml --no-resume
```

## Output Structure

### Individual Model Results
Each model saves its results in the standard TimeFlies output structure:
```
outputs/
├── {project}/
│   ├── experiments/
│   │   ├── uncorrected/
│   │   │   ├── all_runs/
│   │   │   │   └── {tissue}_{model}_{target}/
│   │   │   │       └── {timestamp}/
│   │   │   │           ├── model.h5
│   │   │   │           ├── training/
│   │   │   │           └── evaluation/
│   │   │   └── latest -> all_runs/.../
│   │   └── batch_corrected/
```

### Queue Summary Reports
Queue-specific summaries are saved separately:
```
outputs/
├── model_queue_summaries/
│   ├── {queue_name}_{timestamp}_summary.md
│   └── {queue_name}_{timestamp}_metrics.csv
```

### Summary Report Contents

**Markdown Report (`*_summary.md`):**
- Queue overview and statistics
- Top performing models table
- Detailed results for each model
- Training times and error information

**CSV Report (`*_metrics.csv`):**
- Model name, type, status
- All metrics (accuracy, precision, recall, F1)
- Training time
- All hyperparameters (as `hp_*` columns)
- Easy to import into analysis tools

## Best Practices

### 1. Configuration Design
- **Start small**: Begin with 2-3 models to test your configuration
- **Use descriptive names**: Model names appear in all reports
- **Include descriptions**: Help future you remember what each model tests
- **Organize by purpose**: Group related models together

### 2. Hyperparameter Exploration
- **One variable at a time**: Change one hyperparameter per model for clarity
- **Include baselines**: Always include a baseline model for comparison
- **Use meaningful ranges**: Don't use extreme values unless testing limits

### 3. Preprocessing Comparisons
- **Control variables**: Keep other settings constant when testing preprocessing
- **Document assumptions**: Use descriptions to note why you chose specific settings
- **Test interactions**: Some preprocessing options work better together

### 4. Resource Management
- **Monitor training times**: Start with short epochs for initial testing
- **Use checkpoints**: Always enable checkpoints for long-running queues
- **Plan disk space**: Each model saves full outputs

### 5. Result Analysis
- **Review CSV exports**: Easy to analyze in pandas/R/Excel
- **Compare training times**: Include efficiency in model selection
- **Check for overfitting**: Compare training vs. evaluation metrics

## Troubleshooting

### Common Issues

**Configuration Errors:**
```bash
# Verify configuration syntax
timeflies verify  # Checks for queue configs in system check
```

**Training Failures:**
- Individual model failures don't stop the queue
- Failed models are marked in results with error messages
- Check individual model logs in standard output directories

**Memory Issues:**
- Reduce batch size in hyperparameters
- Limit sample size in data.sampling.samples
- Run models with smaller datasets first

**Disk Space:**
- Each model saves full outputs (models, plots, results)
- Monitor disk usage during long queues
- Consider disabling visualizations for large queues

### Debugging
- Enable verbose output: `timeflies queue config.yaml --verbose`
- Check individual model outputs in standard directories
- Review checkpoint files for progress tracking
- Examine log files in outputs/logs/

## Advanced Usage

### Custom Analysis Scripts
You can integrate custom analysis scripts into the queue:
```yaml
global_settings:
  with_analysis: true
  analysis_script: "templates/my_custom_analysis.py"
```

### Integration with Batch Systems
For large-scale computing, wrap the queue command:
```bash
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --mem=32G

source .activate.sh
timeflies queue configs/large_scale_queue.yaml
```

### Parallel Execution (Future Feature)
```yaml
queue_settings:
  sequential: false  # Enable parallel execution
  max_parallel: 4    # Maximum parallel models
```

The model queue system provides a powerful framework for systematic model comparison and hyperparameter exploration, essential for rigorous machine learning research in computational biology.
