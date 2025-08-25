# TimeFlies Development Roadmap

Transcriptomics aging research computational tool

## 🎉 Recent Milestones

### ✅ Step 1 Complete: Automated Multi-Model Training System (August 2024)
**Major achievement**: Full implementation of automated sequential model training with comprehensive features:
- **ModelQueueManager**: Core system for training 10+ models with different configurations
- **CLI Integration**: `timeflies queue` command with default config support
- **GUI Integration**: "Run Model Queue" button in TimeFlies launcher
- **Advanced Features**: Checkpoint/resume, execution control, configuration overrides
- **Quality**: 12 comprehensive tests (10 unit + 2 e2e), Python 3.12 compatible
- **Documentation**: Complete user guide with examples and best practices
- **Impact**: Enables automated comparative model analysis and hyperparameter exploration

---

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

- [x]  **Build automated training/evaluation queue manager (train 10+ models sequentially)** ✅ COMPLETED
  - ✅ Implemented ModelQueueManager with comprehensive features
  - ✅ Sequential multi-model training with 10+ different configurations
  - ✅ Granular execution control (train-only, eval-only, full workflow)
  - ✅ Checkpoint/resume system for fault-tolerant long-running experiments
  - ✅ Deep configuration merge system for per-model setting overrides
  - ✅ Real-time progress tracking ("X completed, Y remaining")
  - ✅ CLI integration: `timeflies queue` (default config: configs/model_queue.yaml)
  - ✅ GUI integration: "Run Model Queue" button in TimeFlies launcher
  - ✅ System verification: Enhanced `timeflies verify` for queue configs
  - ✅ Comprehensive testing: 10 unit + 2 e2e tests, Python 3.12 compatible
  - ✅ Documentation: Complete user guide at docs/model_queue_guide.md
- [x]  **Create summary reports comparing all model performance** ✅ COMPLETED
  - ✅ Markdown summary reports with top performing models table
  - ✅ CSV metrics export for easy analysis (pandas/R/Excel compatible)
  - ✅ Detailed results with hyperparameters and training times
  - ✅ Training time calculation and performance comparison
- [ ]  Implement hyperparameter search/tuning (grid search, random search, or Bayesian optimization)

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
1. ~~Automated multi-model training system~~ ✅ **COMPLETED** (Step 1)
2. Hyperparameter search/tuning integration (grid search, Bayesian optimization)
3. Bug fixes & documentation cleanup ~~(config.yaml artifact issue resolved)~~
4. Alzheimer's dataset analysis completion (paper deadline)
5. Testing expansion (33% → 75%+)
6. PyPI deployment (research reproducibility)
7. PyTorch migration (major refactor)

**Key Considerations:**

- Maintain backward compatibility during any framework changes
- Ensure reproducible results across different computing environments
- Ensure code design is fully tool-based, making it easy to use with any new dataset (provided the data meets the required structural components)
- after each feature added, update all relvenat documenation, add tests and ensure proper usage.
