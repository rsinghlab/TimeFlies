# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TimeFlies is an ML framework for analyzing single-cell RNA-seq data. It provides deep learning models (CNN, MLP), traditional ML (XGBoost, Random Forest, Logistic Regression), SHAP interpretability, batch correction via scVI-tools, and a complete research CLI + Gradio web GUI.

## Commands

### Install
```bash
# From GitHub
uv pip install git+https://github.com/rsinghlab/TimeFlies

# Development install
git clone https://github.com/rsinghlab/TimeFlies.git && cd TimeFlies
uv pip install -e ".[dev]"

# With batch correction (scvi-tools + PyTorch)
uv pip install -e ".[dev,batch-correction]"
```

### Run the CLI
```bash
timeflies setup                 # Split data + verify + create dirs
timeflies train                 # Train with auto-evaluation
timeflies evaluate              # Evaluate on test data
timeflies queue                 # Multi-model sequential training
timeflies tune                  # Hyperparameter optimization (Optuna)
timeflies gui                   # Launch Gradio web UI
```

### Testing
```bash
pytest tests/                                     # All tests
pytest tests/unit -v                              # Unit tests only
pytest tests/ -m integration                      # By marker
pytest tests/unit/test_config_manager.py -v       # Single file
pytest tests/unit/test_config_manager.py::TestConfigManager::test_load -v  # Single test
pytest tests/ --cov=src --cov-report=term         # With coverage
pytest tests/test_performance.py --benchmark-only # Benchmarks
timeflies test unit --coverage                    # Via CLI wrapper
```

Test markers: `unit`, `integration`, `functional`, `system`, `performance`, `slow`.

Test data uses a 3-tier strategy: **tiny** (50 cells, committed), **synthetic** (500 cells, generated), **real** (5000 cells, gitignored).

### Linting & Formatting
```bash
ruff check src/ tests/          # Lint
ruff check src/ tests/ --fix    # Lint with auto-fix
ruff format src/ tests/         # Format
mypy src/                       # Type check
pre-commit run --all-files      # All pre-commit hooks
```

## Architecture

Source code lives under `src/timeflies/` (the installable package).

```
src/timeflies/
├── cli/              # CLI entry point, argument parser, command dispatch
│   └── commands/     # Per-domain command modules (setup, training, analysis, etc.)
├── core/             # Pipeline orchestration
│   ├── config_manager.py      # YAML config loading/validation
│   ├── pipeline_manager.py    # Main train/eval pipeline orchestrator
│   ├── model_manager.py       # Model lifecycle (create/train/save/load)
│   ├── model_queue.py         # Multi-model sequential training
│   ├── hyperparameter_tuner.py # Optuna-based Bayesian optimization
│   └── analysis_queue.py      # Analysis workflow management
├── data/             # Data loading (AnnData .h5ad), splitting, preprocessing
│   └── preprocessing/         # Gene filtering, batch correction (scVI)
├── models/           # Model definitions (CNN, MLP, XGBoost, RF, Logistic)
│   └── model_factory.py       # Factory pattern for model creation
├── evaluation/       # Metrics calculation, SHAP interpretability
├── analysis/         # EDA, visualization, plot generation
├── gui/              # Gradio web interface launcher
├── display/          # Terminal output formatting
└── utils/            # Path management, GPU handling, logging, constants
```

### Key data flow
1. AnnData `.h5ad` files in `data/[project]/[tissue]/` are loaded via `data/loaders.py`
2. `core/config_manager.py` reads YAML configs from `configs/`
3. `core/pipeline_manager.py` orchestrates: preprocess -> train -> evaluate -> interpret
4. `models/model_factory.py` creates the selected model type
5. Results go to `outputs/[project]/experiments/` with timestamped dirs, `latest`/`best` symlinks

### Configuration
All settings are YAML files in `configs/`:
- `default.yaml` — project, hardware, model type, training hyperparams, preprocessing
- `setup.yaml` — data splitting (ratios, stratification)
- `batch_correction.yaml` — scVI-tools settings (install via `.[batch-correction]` extra)
- `hyperparameter_tuning.yaml` — search algorithm and parameter ranges
- `model_queue.yaml` — multi-model training configurations

## Code Conventions

- **Python 3.12+** required (enforced in pyproject.toml)
- **Ruff** for linting and formatting (line-length 88, target py312)
- Uppercase variable names `X`, `X_train`, `X_test` are intentional ML conventions (ruff N803/N806 ignored)
- `F401` (unused imports) is ignored for `__init__.py` re-exports
- **mypy** with `ignore_missing_imports = true`; `disallow_untyped_defs` is off
- TensorFlow/Keras 3 compatibility — models use `tf.keras` APIs
- Batch correction deps (scvi-tools + PyTorch) installed via `.[batch-correction]` optional extra

## CI/CD

GitHub Actions (`.github/workflows/ci.yml`): uv-based install, tests on Python 3.12 across Ubuntu/macOS/Windows, ruff lint + format check, mypy, bandit security scan, integration tests, and docs deployment.
