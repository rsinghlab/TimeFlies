# TimeFlies CLI Commands

## Core Commands

### `python run_timeflies.py verify`
**Purpose**: System verification before training  
**What it checks**:
- Python environment and packages
- Project configuration
- Data files (original, splits, batch)
- Test fixtures
- Hardware (GPU detection)

**Output**: ✅/❌ status for each component + workflow checklist

---

### `python run_timeflies.py create-test-data` 
**Purpose**: Generate test fixtures from real data  
**What it does**:
- Samples small datasets from your real H5AD files
- Creates fixtures for unit/integration tests
- Generates test data statistics

**Requirements**: Original data files must exist

---

### `python run_timeflies.py setup`
**Purpose**: Create train/eval splits for all projects  
**What it does**:
- Finds all projects with `*_original.h5ad` files
- Creates stratified splits (age/disease + sex + cell type)
- Processes both original and batch-corrected data
- Uses split_size from `configs/setup.yaml`

**Smart features**:
- Never overwrites existing splits
- Identical splits for original and batch data
- Multi-project support

---

### `python run_timeflies.py train`
**Purpose**: Train deep learning models  
**Uses**: Active project from `configs/default.yaml`

---

### `python run_timeflies.py evaluate`
**Purpose**: Evaluate models with SHAP interpretation  
**Creates**: Visualizations and interpretability analysis

---

### `python run_timeflies.py test`
**Purpose**: Run comprehensive test suite  
**Test Types**:
- `unit` - Fast isolated tests (~5-15 seconds)
- `integration` - Component integration (~30-60 seconds) 
- `functional` - Complete workflows (~2-5 minutes)
- `system` - CLI and installation (~10-30 seconds)

**Options**:
- `--fast` - Unit + integration only (quick feedback)
- `--type unit|integration|functional|system` - Run specific test type
- `--coverage` - Generate HTML coverage report
- `--verbose` - Detailed output with test names
- `--debug` - Stop on first failure for debugging
- `--rerun` - Re-run only previously failed tests

**Examples**:
```bash
# Quick feedback while coding
python run_timeflies.py test --fast

# Debug failing test
python run_timeflies.py test --debug --verbose

# Full test suite with coverage
python run_timeflies.py test --coverage

# Re-run just the failures
python run_timeflies.py test --rerun
```

---

## Batch Correction Workflow

```bash
# 1. Setup batch environment (one-time)
bash setup_dev_env.sh  # choose 'y'

# 2. Activate batch environment
source activate_batch.sh

# 3. Run batch correction
python run_timeflies.py batch-correct

# 4. Return to main environment  
source activate.sh

# 5. Create batch splits
python run_timeflies.py setup
```

## Configuration

**Switch projects**: Edit `configs/default.yaml`
```yaml
project: fruitfly_aging  # or fruitfly_alzheimers
```

**Split parameters**: Edit `configs/setup.yaml`
```yaml
data:
  train_test_split:
    split_size: 5000    # cells in eval set
general:
  random_state: 42      # reproducible splits
```