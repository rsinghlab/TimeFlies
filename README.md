# TimeFlies

A single-nucleus RNA-seq aging clock for the *Drosophila melanogaster* head, built on a 1D convolutional neural network.

**Paper:** Tennant, Pavuluri, Singh, Cortez, O'Connor-Giles, Larschan & Singh. *Scientific Reports* (2026). *(in press)*

## Installation

```bash
uv pip install git+https://github.com/rsinghlab/TimeFlies

# With batch correction support (scvi-tools + PyTorch)
uv pip install "timeflies[batch-correction] @ git+https://github.com/rsinghlab/TimeFlies"
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Quick Start

### Command line

```bash
# 1. Place *_original.h5ad files in data/[project]/[tissue]/
# 2. Edit configs/default.yaml

timeflies setup          # create holdout evaluation set, create directories
timeflies train          # train model with automatic evaluation
timeflies evaluate       # evaluate on holdout set
```

### Python

```python
import timeflies

results = timeflies.train()                    # uses configs/default.yaml
results = timeflies.train("my_config.yaml")    # custom config
metrics = timeflies.evaluate()
adata   = timeflies.load_data("data/cells.h5ad")
```

All CLI commands are available as Python functions: `setup()`, `train()`, `evaluate()`, `eda()`, `analyze()`, `split()`, `batch_correct()`, `tune()`, `queue()`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `timeflies setup` | Split data and create directories |
| `timeflies train` | Train model (includes automatic evaluation) |
| `timeflies evaluate` | Evaluate trained model on test set |
| `timeflies analyze` | Run project-specific analysis scripts |
| `timeflies eda` | Exploratory data analysis |
| `timeflies split` | Create train/eval data splits only |
| `timeflies batch-correct` | Apply scVI batch correction (requires `[batch-correction]` extra) |
| `timeflies tune` | Hyperparameter optimization (grid, random, or Bayesian) |
| `timeflies queue` | Sequential multi-model training |
| `timeflies test` | Run test suite |
| `timeflies create-test-data` | Generate test fixtures |

Global flags: `--verbose`, `--batch-corrected`, `--tissue`, `--model`, `--target`, `--project`

## Configuration

All settings live in `configs/`:

| File | Purpose |
|------|---------|
| `default.yaml` | Project, model, data paths, and training hyperparameters |
| `setup.yaml` | Holdout evaluation set creation (size, stratification) |
| `batch_correction.yaml` | scVI batch correction settings |

Example configs for hyperparameter tuning and model queues are in `examples/`.

## Data Format

TimeFlies expects [AnnData](https://anndata.readthedocs.io/) `.h5ad` files:

```
data/
  fruitfly_aging/
    head/
      drosophila_head_aging_original.h5ad    # raw input
      drosophila_head_aging_train.h5ad       # created by timeflies setup
      drosophila_head_aging_eval.h5ad        # created by timeflies setup
```

The AFCA dataset used in the paper contains 289,981 cells across 15,992 genes with four age classes (Day 5, 30, 50, 70). The model uses the entire transcriptome without feature selection.

## Models

The primary model is a 1D CNN classifying donor age from genome-wide expression profiles. Benchmarking models are also provided: XGBoost, Random Forest, MLP, and Logistic Regression. SHAP values (GradientExplainer) are available for interpretability.

## About

TimeFlies was originally developed as an snRNA-seq aging clock for *Drosophila melanogaster* head tissue. Aging biomarker genes identified by the clock were cross-validated against differentially expressed genes in Alzheimer's disease fly models (ADFCA). The underlying framework has since been adapted to be more general-purpose and can be applied to any single-cell classification task with AnnData input.

## Citation

```bibtex
@article{tennant2026timeflies,
  title     = {An {snRNA}-seq aging clock for the fruit fly head sheds light
               on sex-biased aging},
  author    = {Tennant, Nikolai and Pavuluri, Ananya and Singh, Gunjan and
               Cortez, Kaitlyn and O'Connor-Giles, Kate and Larschan, Erica and
               Singh, Ritambhara},
  journal   = {Scientific Reports},
  year      = {2026},
  note      = {In press}
}
```

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE](LICENSE) for terms.
