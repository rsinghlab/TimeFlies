# Hyperparameter Tuning Guide

TimeFlies includes comprehensive hyperparameter tuning capabilities with three optimization methods: grid search, random search, and Bayesian optimization using Optuna.

## Quick Start

### 1. Enable Hyperparameter Tuning

Edit your `configs/default.yaml` and set:

```yaml
hyperparameter_tuning:
  enabled: true  # Enable hyperparameter optimization
  method: "bayesian"  # Options: "grid", "random", "bayesian"
  n_trials: 20
```

### 2. Run Hyperparameter Tuning

```bash
# Use your existing default.yaml configuration
timeflies tune

# Or specify a custom config file
timeflies tune configs/my_custom_config.yaml
```

### 3. Check Results

Results are saved in `outputs/hyperparameter_tuning/` with:
- **Comprehensive markdown report** with top performing models
- **CSV metrics export** for analysis in pandas/R/Excel
- **Resume checkpoints** for interrupted searches

## Configuration

Hyperparameter tuning is configured directly in your `configs/default.yaml`. This eliminates duplication and uses your existing project settings as the base.

### Basic Configuration Structure

```yaml
hyperparameter_tuning:
  enabled: true
  method: "bayesian"  # "grid", "random", or "bayesian"
  n_trials: 20        # For random/bayesian (ignored for grid)

  # Speed optimizations for hyperparameter search
  search_optimizations:
    data:
      sampling:
        samples: 1000    # Use subset for faster trials
        variables: 500   # Use top genes for speed
    with_eda: false      # Skip EDA during search
    with_analysis: false # Skip analysis during search
    interpret: false     # Skip SHAP during search
    model:
      training:
        epochs: 50       # Reduced epochs for search
        early_stopping_patience: 5

  # Define hyperparameters to tune for each model type
  model_hyperparams:
    CNN:
      learning_rate: [0.0001, 0.001, 0.01]
      batch_size: [16, 32, 64]
      epochs: [50, 75, 100]

      # CNN architecture variants
      cnn_variants:
        - name: "standard"
          filters: [32]
          kernel_sizes: [3]
          pool_sizes: [2]
        - name: "larger_filters"
          filters: [64]
          kernel_sizes: [3]
          pool_sizes: [2]

    xgboost:
      n_estimators: [100, 200, 300]
      max_depth: [6, 9, 12]
      learning_rate: [0.01, 0.1, 0.2]
```

## Search Methods

### Grid Search
- **Best for**: Small parameter spaces, comprehensive exploration
- **Explores**: All possible combinations systematically
- **Use when**: You have few parameters and want to be thorough

```yaml
hyperparameter_tuning:
  method: "grid"
  # n_trials ignored - explores all combinations
```

### Random Search
- **Best for**: Larger parameter spaces, time-constrained searches
- **Explores**: Random sampling of parameter combinations
- **Use when**: You have many parameters and limited time

```yaml
hyperparameter_tuning:
  method: "random"
  n_trials: 50  # Number of random samples
```

### Bayesian Optimization (Optuna)
- **Best for**: Intelligent optimization, learning from previous trials
- **Explores**: Smart parameter selection based on past results
- **Use when**: You want the most efficient hyperparameter search

```yaml
hyperparameter_tuning:
  method: "bayesian"
  n_trials: 30  # Usually needs fewer trials than random
```

## CNN Architecture Variants

For CNN models, you can explore different architectures along with hyperparameters:

```yaml
model_hyperparams:
  CNN:
    learning_rate: [0.001, 0.01]
    batch_size: [16, 32, 64]

    # Architecture variants based on your existing CNN structure
    cnn_variants:
      - name: "lightweight"
        filters: [16]        # Smaller filters
        kernel_sizes: [3]
        pool_sizes: [2]

      - name: "standard"
        filters: [32]        # Your current default
        kernel_sizes: [3]
        pool_sizes: [2]

      - name: "larger_filters"
        filters: [64]        # More filters
        kernel_sizes: [3]
        pool_sizes: [2]

      - name: "larger_kernel"
        filters: [32]
        kernel_sizes: [5]    # Larger receptive field
        pool_sizes: [null]   # No pooling
```

Each variant is combined with all hyperparameter combinations.

## Integration with Model Queue

After hyperparameter tuning, use the best configurations for production training:

```python
from common.core.model_queue import ModelQueueManager

# Create a model queue from hyperparameter results
manager = ModelQueueManager.from_hyperparameter_results(
    hyperparameter_results_dir="outputs/hyperparameter_tuning/search_2024-08-25_16-30-45",
    top_n=5  # Use top 5 configurations
)

# Run production training with full analysis
manager.run_production_training(
    enable_full_analysis=True,
    enable_interpretation=True
)
```

## Advanced Usage

### Resume Interrupted Searches

Hyperparameter searches automatically save checkpoints:

```bash
# Resumes from checkpoint if available
timeflies tune

# Force fresh start
timeflies tune --no-resume
```

### Custom Parameter Ranges

You can define any hyperparameter that your model accepts:

```yaml
model_hyperparams:
  CNN:
    # Training hyperparameters
    learning_rate: [0.0001, 0.001, 0.01]
    batch_size: [16, 32, 64]
    epochs: [25, 50, 75, 100]

    # Early stopping
    early_stopping_patience: [5, 8, 10]

    # Optimizer settings
    optimizer: ["adam", "sgd", "rmsprop"]

    # Model architecture (for variants)
    dropout_rate: [0.2, 0.3, 0.5]
```

### Multiple Model Types

Configure hyperparameters for different model types:

```yaml
model_hyperparams:
  CNN:
    learning_rate: [0.001, 0.01]
    batch_size: [16, 32, 64]

  xgboost:
    n_estimators: [100, 200, 300]
    max_depth: [6, 9, 12]
    learning_rate: [0.01, 0.1, 0.2]

  random_forest:
    n_estimators: [100, 200]
    max_depth: [10, 20, null]
    min_samples_split: [2, 5]
```

Only the model type specified in `data.model` will be tuned.

## Output Structure

Hyperparameter tuning results are organized in timestamped directories:

```
outputs/hyperparameter_tuning/
└── timeflies_hyperparameter_search_2024-08-25_16-30-45/
    ├── hyperparameter_search_report.md    # Comprehensive results report
    ├── hyperparameter_search_metrics.csv  # Metrics for all trials
    ├── checkpoint.json                     # Resume checkpoint
    ├── search_config.yaml                  # Configuration backup
    └── optuna_study.db                     # Bayesian optimization database
```

### Results Report

The markdown report includes:
- **Best trial** with optimal hyperparameters
- **Top 5 trials** comparison table
- **Search configuration** used
- **Performance metrics** for all completed trials

### Metrics CSV

The CSV export contains:
- Trial index and status
- All hyperparameters (`param_*` columns)
- Architecture settings (`arch_*` columns)
- Performance metrics (accuracy, precision, recall, F1)
- Training time and timestamps

Perfect for analysis in pandas, R, or Excel.

## Best Practices

### 1. Start Small
Begin with a small parameter space and short training times:

```yaml
search_optimizations:
  data:
    sampling:
      samples: 500    # Small subset first
      variables: 100
  model:
    training:
      epochs: 25      # Short training
      early_stopping_patience: 3
```

### 2. Use Bayesian Optimization
For most cases, Bayesian optimization is the most efficient:

```yaml
hyperparameter_tuning:
  method: "bayesian"
  n_trials: 20  # Often sufficient for good results
```

### 3. Monitor Progress
Hyperparameter search shows real-time progress:

```
🔄 Running hyperparameter trial 5/20
   Variant: cnn_standard
   Parameters: {'learning_rate': 0.001, 'batch_size': 32}
✅ Trial 5 completed in 45.2s
   Metrics: {'accuracy': 0.847, 'f1_score': 0.834}
📊 Progress: 5/20 trials completed, 15 remaining
```

### 4. Production Training
After finding optimal hyperparameters, run production training with full analysis:

1. **Hyperparameter search**: Fast exploration with reduced features
2. **Production training**: Full training with optimal parameters + SHAP + visualizations

## Troubleshooting

### "Hyperparameter tuning is not enabled"
Set `hyperparameter_tuning.enabled: true` in your config file.

### "No hyperparameters defined for model type"
Add hyperparameters for your model type in the `model_hyperparams` section.

### Out of Memory
Reduce the dataset size during search:

```yaml
search_optimizations:
  data:
    sampling:
      samples: 200    # Very small for memory-constrained systems
      variables: 50
```

### Optuna Import Error
Install Optuna for Bayesian optimization:

```bash
pip install optuna>=3.0.0
```

## Examples

### Quick CNN Optimization
```yaml
hyperparameter_tuning:
  enabled: true
  method: "bayesian"
  n_trials: 15
  model_hyperparams:
    CNN:
      learning_rate: [0.001, 0.01]
      batch_size: [16, 32, 64]
      epochs: [50, 75]
```

### Comprehensive Architecture Search
```yaml
hyperparameter_tuning:
  enabled: true
  method: "grid"
  model_hyperparams:
    CNN:
      learning_rate: [0.001, 0.01]
      batch_size: [32, 64]
      cnn_variants:
        - name: "small"
          filters: [16]
          kernel_sizes: [3]
        - name: "medium"
          filters: [32]
          kernel_sizes: [3]
        - name: "large"
          filters: [64]
          kernel_sizes: [5]
```

This creates 2 × 2 × 3 = 12 total combinations to explore.

## Integration with TimeFlies Workflow

Hyperparameter tuning integrates seamlessly with your existing workflow:

1. **Setup**: Use your normal `timeflies setup` command
2. **Tune**: Run `timeflies tune` for hyperparameter optimization
3. **Production**: Use best parameters for final training with `timeflies train`

The hyperparameter search uses all your existing settings (project, data paths, preprocessing) but optimizes the model parameters for best performance.
