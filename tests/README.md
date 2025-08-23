# TimeFlies v1.0 Test Suite

Clean, organized tests with fast feedback loops and 3-tier test data strategy.

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
timeflies test [unit|integration|functional|system|all] [--coverage] [--fast] [--debug] [--rerun]

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

The test runner automatically handles batch correction tests:

**What happens when you run `timeflies test --coverage`:**
1. 🧪 **Regular tests** run in main `.venv` environment
2. 🔄 **Auto-detects** batch correction tests  
3. 🧬 **Switches** to `.venv_batch` environment for batch correction tests
4. 🔄 **Returns** to main `.venv` environment  
5. 📊 **Combines** coverage reports from both environments

**Developer Experience:**
- ✅ **One command**: `timeflies test` - everything works automatically
- ✅ **No manual switching**: Test runner handles environments  
- ✅ **Always ends in main env**: Never left in wrong environment
- ✅ **Comprehensive testing**: Real batch correction testing, not just mocks

## Test Data Tiers

**Tiny** (50 cells, 100 genes): Committed to git, always available  
**Synthetic** (500 cells, 1000 genes): Generated from metadata, realistic patterns  
**Real** (5000 cells, 2000 genes): Gitignored, full complexity testing  

```bash
# Check existing fixtures: ls tests/fixtures/*/
# Generate missing fixtures: timeflies create-test-data --tier [tiny|synthetic|real]
```