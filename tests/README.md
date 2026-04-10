# Tests

## Running Tests

```bash
# All tests
timeflies test

# By category
timeflies test unit
timeflies test integration
timeflies test functional
timeflies test system

# With coverage
timeflies test --coverage

# Quick feedback (unit + integration only)
timeflies test --fast

# Re-run only previously failed tests
timeflies test --rerun

# Or use pytest directly
pytest tests/ -v
pytest tests/unit/ -v
pytest tests/test_performance.py --benchmark-only -v
```

## Markers

| Marker         | Scope                                    |
|----------------|------------------------------------------|
| `unit`         | Fast isolated component tests            |
| `integration`  | Cross-component tests with synthetic data|
| `functional`   | Full end-to-end workflows                |
| `system`       | CLI and installation validation          |
| `performance`  | Benchmarks (data, models, CLI, memory)   |
| `slow`         | Long-running tests                       |

## Test Data (3-Tier)

| Tier      | Size                  | Location                          |
|-----------|-----------------------|-----------------------------------|
| Tiny      | 50 cells, 100 genes   | `tests/fixtures/` (committed)     |
| Synthetic | 500 cells, 1000 genes | Generated: `timeflies create-test-data --tier synthetic` |
| Real      | 5000 cells, 2000 genes| Generated locally (gitignored)    |
