# TimeFlies

A single-nucleus RNA-seq aging clock for the *Drosophila melanogaster* head, built on a 1D convolutional neural network.

**Paper:** Tennant, Pavuluri, Singh, Cortez, O'Connor-Giles, Larschan & Singh. *Nature Scientific Reports* (2026).
[https://doi.org/10.1038/s41598-026-48613-0](https://doi.org/10.1038/s41598-026-48613-0)

## Installation

```bash
uv pip install git+https://github.com/rsinghlab/TimeFlies

# With batch correction support (scvi-tools + PyTorch)
uv pip install "timeflies[batch-correction] @ git+https://github.com/rsinghlab/TimeFlies"
```

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

## Quick Start

```bash
# 1. Place *_original.h5ad files in data/[project]/[tissue]/
# 2. Edit configs/setup.yaml and configs/default.yaml

timeflies setup          # split data, create directories, verify system
timeflies train          # train model with automatic evaluation
timeflies evaluate       # evaluate on held-out test data
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `timeflies setup` | Split data + create directories + system check |
| `timeflies train` | Train model (includes automatic evaluation) |
| `timeflies evaluate` | Evaluate trained model on test set |
| `timeflies analyze` | Run project-specific analysis scripts |
| `timeflies eda` | Exploratory data analysis |
| `timeflies split` | Create train/eval data splits only |
| `timeflies batch-correct` | Apply scVI batch correction (requires `[batch-correction]` extra) |
| `timeflies tune` | Hyperparameter optimization (grid, random, or Bayesian) |
| `timeflies queue` | Sequential multi-model training |
| `timeflies verify` | System and environment check |
| `timeflies test` | Run test suite |
| `timeflies create-test-data` | Generate test fixtures |

Global flags: `--verbose`, `--batch-corrected`, `--tissue`, `--model`, `--target`, `--project`

## Configuration

All settings live in `configs/`:

| File | Purpose |
|------|---------|
| `default.yaml` | Project, model, data paths, and training hyperparameters |
| `setup.yaml` | Data splitting (split ratio, stratification) |
| `batch_correction.yaml` | scVI batch correction settings |
| `hyperparameter_tuning.yaml` | Search spaces for `timeflies tune` |
| `model_queue.yaml` | Model sequence for `timeflies queue` |

## Data Format

TimeFlies expects [AnnData](https://anndata.readthedocs.io/) `.h5ad` files:

```
data/
  fruitfly_aging/
    head/
      head_original.h5ad    # raw input
      head_train.h5ad       # created by timeflies setup
      head_eval.h5ad        # created by timeflies setup
```

The AFCA dataset used in the paper contains 289,981 cells across 15,992 genes with four age classes (Day 5, 30, 50, 70). The model uses the entire transcriptome without feature selection.

## Models

The primary model is a 1D CNN classifying donor age from genome-wide expression profiles. Benchmarking models are also provided: XGBoost, Random Forest, MLP, and Logistic Regression. SHAP values (GradientExplainer) are available for interpretability.

## Citation

```bibtex
@article{tennant2026timeflies,
  title     = {TimeFlies: a single-nucleus {RNA}-seq aging clock for the
               {Drosophila melanogaster} head},
  author    = {Tennant, Nikolai and Pavuluri, Akhil and Singh, Aaditya and
               Cortez, Alain and O'Connor-Giles, Kate and Larschan, Erica and
               Singh, Ritambhara},
  journal   = {Nature Scientific Reports},
  year      = {2026},
  doi       = {10.1038/s41598-026-48613-0}
}
```

## License

[CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). See [LICENSE](LICENSE) for terms.
