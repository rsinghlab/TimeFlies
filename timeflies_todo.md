# TimeFlies Development Roadmap

Transcriptomics aging research computational tool

## 🔬 Pre-Publication Critical Tasks

### Testing & Quality Assurance

- **Complete testing suite** - Target 75%+ coverage (currently 33%)
- [ ]  Add more tests for core analysis functions
- [ ]  Integration tests for data pipelines
- [ ]  Edge case handling tests
- **Code cleanup & documentation audit**
- [ ]  Remove deprecated/unused code dependencies
- [ ]  Update all docstrings to match current functionality
- [ ]  Verify all functions have proper type hints
- [ ]  Code style consistency check (ruff)
- **Bug fixes** - Address all known issues before publication

### Deployment & Distribution

- **PyPI package preparation**
- [ ]  Update [setup.py/pyproject.toml](http://setup.py/pyproject.toml) with correct metadata
- [ ]  Verify all dependencies are properly specified (especially new PyTorch deps)
- [ ]  Test installation from PyPI test repository
- [ ]  Update existing README for any setup/installation changes
- **Comprehensive stress testing**
- [ ]  Large dataset performance benchmarks
- [ ]  Memory usage profiling
- [ ]  AI-generated code validation and review!!!
- [ ]  Cross-platform compatibility testing

## 🚀 Feature Enhancements

### Automation & Workflow

- [ ]  Build automated training/evaluation queue manager (train 10+ models sequentially)
- [ ]  Implement hyperparameter search/tuning (grid search, random search, or Bayesian optimization)
- [ ]  Create summary reports comparing all model performance

### Technical Infrastructure

- **PyTorch migration**
- [ ]  Plan conversion strategy from current framework
- [ ]  Convert core model architecture to PyTorch
- [ ]  Update training loops and optimization
- [ ]  Migrate data loaders and preprocessing
- [ ]  Validate results match original implementation
- **Environment standardization**
- [ ]  Finalize containerization approach (Docker vs Conda vs venv (current))
- [ ]  Update dependencies for PyTorch ecosystem

## 📊 Research-Specific Tasks

### Alzheimer's Dataset Analysis

- **Execute batch correction on Alzheimer's datasets**
- [ ]  Run existing batch correction pipeline on target data
- [ ]  Validate correction results and quality metrics
- **Alzheimer's model training & analysis**
- [ ]  Execute full training pipeline with corrected data
- [ ]  Generate comprehensive results for Ananya
- [ ]  Prepare visualizations and statistical summaries

## 📋 Project Management Notes

- **Priority Order:**
1. Automated multi-model training system + hyperparameter tuning
2. Bug fixes & documentation cleanup ( i have noticed config.yaml is created in main direcotry. that shouldnt be there.)
3. Alzheimer's dataset analysis completion (paper deadline)
4. Testing expansion (33% → 75%+)
5. PyPI deployment (research reproducibility)
6. PyTorch migration (major refactor)

**Key Considerations:**

- Maintain backward compatibility during any framework changes
- Ensure reproducible results across different computing environments
- Ensure code design is fully tool-based, making it easy to use with any new dataset (provided the data meets the required structural components)
- after each feature added, update all relvenat documenation, add tests and ensure proper usage.
