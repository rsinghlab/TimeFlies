# TimeFlies Test Suite

Comprehensive test suite with performance benchmarking, system validation, and production-ready testing.

## Recent Improvements ✨

- **Performance Benchmarks**: 7 benchmarks covering data, models, CLI, and memory usage
- **System Tests**: 15+ tests for installation validation and environment checking
- **Improved Organization**: Renamed files for clarity and better structure
- **Real Data Testing**: Minimal mocking, maximum real functionality testing
- **Production Ready**: Complete CI/CD integration with automated performance monitoring

## Test Organization

### 📁 Unit Tests (`tests/unit/`) - Component-level testing
- `test_eda_analysis.py` - EDA and analysis functionality
- `test_cli_commands.py` - CLI command execution
- `test_cli_parser.py` - CLI argument parsing
- `test_configuration.py` - Configuration management
- `test_data_processing.py` - Data preprocessing pipelines
- `test_evaluation_metrics.py` - Metrics and evaluation
- `test_evaluation_integration.py` - Evaluation workflow integration
- `test_integration_workflows.py` - Multi-component workflows
- `test_model_factory.py` - Model creation and management
- `test_model_training.py` - Model training processes
- `test_pipeline_components.py` - Pipeline component testing
- `test_pipeline_manager.py` - Pipeline orchestration

### 📁 System Tests (`tests/system/`) - Environment validation
- `test_installation.py` - Installation validation, environment checking, system setup

### 📁 Performance Tests (`tests/`) - Benchmarking
- `test_performance.py` - Performance benchmarks for data, models, CLI, and memory

### 📁 Integration Tests (`tests/integration/`) - Cross-component testing
- `test_real_data_workflows.py` - Real data processing workflows

### 📁 Functional Tests (`tests/functional/`) - End-to-end user workflows
- Complete workflow testing from user perspective
- Full command execution: setup → train → evaluate → analyze
- Batch correction workflows with automatic environment switching
- Multi-project testing scenarios

## Performance Benchmarking 🚀

New performance test suite with 7 benchmarks:
```bash
# Run performance benchmarks
pytest tests/test_performance.py --benchmark-only -v

# Example results:
# AnnData creation: ~37ms for 1000x500 matrix
# Data loading: ~4.5ms for 500x200 dataset
# sklearn training: ~36ms for RandomForest
# Prediction: ~179µs for 100 samples
# CLI help: ~1.8ms (fast help system)
```

## System Validation 🔧

System tests validate production readiness:
```bash
pytest tests/system/ -v

# Tests include:
# - Python version compatibility (3.10+)
# - Core dependency availability
# - Package structure integrity
# - CLI functionality validation
# - Configuration system testing
# - Environment constraints (memory, disk, GPU)
```

## Structure

```
tests/
├── unit/          # Fast isolated tests (~5-15 seconds)
├── integration/   # Component integration (~30-60 seconds)
├── functional/    # Complete workflows (~2-5 minutes)
├── system/        # CLI and installation (~10-30 seconds)
└── fixtures/      # 3-Tier Test Data Strategy
    ├── fruitfly_aging/
    │   ├── tiny_head.h5ad              # Tier 1: 50 cells, 100 genes (committed)
    │   ├── tiny_head_batch.h5ad        # Batch-corrected version
    │   ├── synthetic_head.h5ad         # Tier 2: 500 cells, 1000 genes (generated)
    │   ├── synthetic_head_batch.h5ad   # Batch-corrected version
    │   └── test_data_head_stats.json   # Metadata for generation
    └── fruitfly_alzheimers/           # Same structure for disease models
```

## Commands

```bash
# Run tests (see main README for all CLI commands)
timeflies test [unit|integration|functional|system|all] [--coverage] [--verbose] [--fast] [--debug] [--rerun]

# Generate test data for testing
timeflies create-test-data --tier tiny --batch-versions      # (optional - already committed)
timeflies create-test-data --tier synthetic --batch-versions # Generated on demand
timeflies create-test-data --tier real --batch-versions      # Local performance testing
```

## 3-Tier Test Data Strategy

**Tier 1 - Tiny** (50 cells, 100 genes): Committed to git, fast CI/CD
- Use for: Unit tests, fast feedback, continuous integration
- Files: `tiny_*.h5ad` (always available)

**Tier 2 - Synthetic** (500 cells, 1000 genes): Generated from metadata
- Use for: Integration tests, realistic biological patterns
- Files: `synthetic_*.h5ad` (generated on demand)

**Tier 3 - Real** (5000 cells, 2000 genes): Large development samples
- Use for: Performance testing, full biological complexity
- Files: `real_*.h5ad` (gitignored, created locally)

## What Each Type Tests

**Unit**: Individual functions with mocks and tiny fixtures - no training
**Integration**: Components working together with synthetic data
- Real biological patterns but manageable size
- Data loading, preprocessing, model creation
**Functional**: Complete end-to-end workflows with full training
- Full pipeline from data → model → evaluation → analysis
**System**: CLI commands and installation validation
- Command parsing, environment setup, error handling

## Testing Workflow

```bash
# Quick development feedback
timeflies test --fast         # Unit + integration only (auto-handles environments)

# Full test suite before commit
timeflies test --coverage     # All tests + coverage report (auto-switches environments)

# Debug specific failures
timeflies test --debug unit   # Stop on first unit test failure
timeflies test --rerun        # Re-run only failed tests
```

## Automatic Environment Switching

TimeFlies automatically manages environments for batch correction across all modes:

**Testing (`timeflies test --coverage`):**
1. Regular tests run in main `.venv` environment
2. Auto-detects batch correction tests requiring scVI dependencies
3. Switches to `.venv_batch` environment for batch correction tests
4. Returns to main `.venv` environment
5. Combines coverage reports from both environments

**User Commands:**
- `timeflies batch-correct` → Auto-switches to batch env, runs correction, exits
- `timeflies setup --batch-correct` → Auto-switches during batch correction step
- `timeflies train --batch-corrected` → Uses existing batch-corrected files (main env)

**Functional Test Coverage:**
- Complete setup → batch-correct → train → evaluate workflows
- Per-project batch correction enable/disable testing
- Environment switching validation
- Multi-project scenario testing

**Developer Experience:**
- One command testing: `timeflies test` handles everything automatically
- No manual environment switching required
- Always returns to main environment
- Real batch correction testing with proper dependencies

## Test Data Tiers

**Tiny** (50 cells, 100 genes): Committed to git, always available
**Synthetic** (500 cells, 1000 genes): Generated from metadata, realistic patterns
**Real** (5000 cells, 2000 genes): Gitignored, full complexity testing

```bash
# Check existing fixtures: ls tests/fixtures/*/
# Generate missing fixtures: timeflies create-test-data --tier [tiny|synthetic|real]
```
