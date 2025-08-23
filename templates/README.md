# TimeFlies Custom Analysis Templates

This directory contains template scripts for creating custom analysis workflows that extend TimeFlies' built-in capabilities.

## Quick Start

1. **Copy a template**: Choose the template closest to your needs
2. **Customize the analysis**: Modify the `run_analysis` function
3. **Run your analysis**: `timeflies analyze --analysis-script path/to/your_script.py`

## Available Templates

### `custom_analysis_example.py`
Basic template showing all available functionality:
- Access to trained model
- Configuration settings
- Predictions data
- File path management
- Custom output generation

**Best for**: First-time users, general analysis tasks

### `aging_analysis_template.py`
Specialized template for aging research:
- Aging acceleration analysis
- Age group performance comparison
- Temporal pattern analysis
- Biomarker discovery workflows

**Best for**: Aging research, longitudinal studies

## Template Structure

All templates must include a `run_analysis` function:

```python
def run_analysis(model, config, path_manager, pipeline):
    """
    Required function that TimeFlies will call.
    
    Args:
        model: Trained model instance (may be None)
        config: Full configuration object
        path_manager: PathManager for file operations
        pipeline: Complete PipelineManager instance
    """
    # Your analysis code here
    pass
```

## Available Objects

### `model`
The trained model instance (if available):
```python
if model is not None:
    print(f"Model type: {type(model).__name__}")
    if hasattr(model, 'predict'):
        predictions = model.predict(X)
```

### `config`
Full configuration with project settings:
```python
project = getattr(config, 'project', 'unknown')
tissue = config.data.tissue
model_type = config.data.model
target = config.data.target_variable
```

### `path_manager`
File path management utilities:
```python
experiment_dir = path_manager.get_experiment_dir()
predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
```

### `pipeline`
Complete pipeline manager with access to data:
```python
if hasattr(pipeline, 'adata_eval'):
    adata = pipeline.adata_eval  # Evaluation data
    print(f"Data: {adata.n_obs} cells, {adata.n_vars} genes")
```

## Common Analysis Patterns

### Load Predictions
```python
predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
if predictions_file.exists():
    predictions_df = pd.read_csv(predictions_file)
    # Analyze predictions...
```

### Access Raw Data
```python
if hasattr(pipeline, 'adata_eval') and pipeline.adata_eval is not None:
    adata = pipeline.adata_eval
    # Analyze single-cell data...
```

### Save Custom Results
```python
analysis_dir = Path(experiment_dir) / "custom_analysis"
analysis_dir.mkdir(exist_ok=True)

results = {"your_metric": value}
with open(analysis_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)
```

### Create Visualizations
```python
plt.figure(figsize=(10, 6))
plt.scatter(x, y)
plt.savefig(analysis_dir / "plot.png", dpi=300, bbox_inches='tight')
plt.close()
```

## Integration with TimeFlies

Your custom analysis runs as part of the TimeFlies workflow:

```bash
# Run with predictions from trained model
timeflies analyze --analysis-script your_analysis.py

# Run with specific predictions file
timeflies analyze --analysis-script your_analysis.py --predictions-path path/to/predictions.csv

# Run with project override
timeflies --aging analyze --analysis-script your_analysis.py
```

## Best Practices

1. **Handle exceptions gracefully**: Wrap analysis in try-catch blocks
2. **Check data availability**: Always verify data exists before processing
3. **Save intermediate results**: Help with debugging and reproducibility
4. **Use descriptive output**: Print progress and results clearly
5. **Create organized output**: Use subdirectories for different analysis types

## Example Custom Analysis Workflow

```python
def run_analysis(model, config, path_manager, pipeline):
    # 1. Setup
    experiment_dir = path_manager.get_experiment_dir()
    analysis_dir = Path(experiment_dir) / "my_analysis"
    analysis_dir.mkdir(exist_ok=True)
    
    # 2. Load data
    predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
    predictions_df = pd.read_csv(predictions_file)
    
    # 3. Run analysis
    results = my_custom_analysis(predictions_df)
    
    # 4. Save results
    with open(analysis_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # 5. Create plots
    create_custom_plots(results, analysis_dir)
    
    print("✅ Custom analysis completed!")
```

## Getting Help

- Check the example templates for common patterns
- TimeFlies configuration: `configs/default.yaml`
- Pipeline source: `src/common/core/pipeline_manager.py`
- Built-in analysis: `src/analysis/`

## Contributing

Create new templates for common analysis patterns and submit them via pull request!