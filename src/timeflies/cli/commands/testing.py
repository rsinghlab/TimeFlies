"""
TimeFlies CLI Testing Commands

Contains test suite execution, test data creation, and all test data helper functions.
"""

import os
import subprocess
import sys
from pathlib import Path


def run_system_tests(args) -> int:
    """Run test suite or provide helpful guidance for users."""
    # Check if tests directory exists (developer installation)
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("TimeFlies Test Information")
        print("=" * 50)
        print("WARNING: Full test suite not available in user installation")
        print("")
        print("To verify your installation works:")
        print("   timeflies verify")
        print("")
        print("For full development testing:")
        print("   git clone https://github.com/rsinghlab/TimeFlies.git")
        print("   cd TimeFlies")
        print("   timeflies test --coverage")
        print("")
        print("NOTE: User installation is working correctly if:")
        print("   • timeflies verify passes")
        print("   • timeflies setup completes successfully")
        print("   • Your research workflow runs without errors")
        return 0

    # Developer path - run actual tests
    print("TESTING: Running TimeFlies Test Suite...")
    print("=" * 50)

    # Build test command
    cmd = [sys.executable, "-m", "pytest", "tests/"]

    # Add test type filter
    test_type = getattr(args, "test_type", "all")
    if test_type != "all":
        cmd.extend(["-m", test_type])

    # Add options
    if getattr(args, "coverage", False):
        cmd.extend(["--cov=src", "--cov-report=html", "--cov-report=term"])

    if getattr(args, "verbose", False):
        cmd.append("-v")

    if getattr(args, "fast", False):
        cmd.extend(["-m", "not functional and not system"])

    if getattr(args, "debug", False):
        cmd.extend(["-x", "-v", "-s"])

    if getattr(args, "rerun", False):
        cmd.append("--lf")  # last failed

    try:
        result = subprocess.run(cmd, cwd=Path.cwd())
        return result.returncode
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return 1


def run_test_suite(args) -> int:
    """Run the test suite using test_runner.py functionality."""
    print("Running TimeFlies Test Suite...")
    print("=" * 60)

    try:
        # Import the test runner function
        sys.path.append(str(Path.cwd() / "tests"))
        from test_runner import run_tests

        # Extract test options from args
        test_type = getattr(args, "test_type", "all")
        coverage = getattr(args, "coverage", False)
        fast = getattr(args, "fast", False)
        debug = getattr(args, "debug", False)
        rerun = getattr(args, "rerun", False)
        verbose = getattr(args, "verbose", False)

        print(f"Test type: {test_type}")
        if fast:
            print("Mode: Fast (unit + integration only)")
        if coverage:
            print("Coverage: Enabled")
        if debug:
            print("Debug: Stop on first failure")
        if rerun:
            print("Re-run: Failed tests only")
        print("")

        # Run the tests
        success = run_tests(
            test_type=test_type,
            verbose=verbose,
            coverage=coverage,
            fast=fast,
            debug=debug,
            rerun_failures=rerun,
        )

        return 0 if success else 1

    except ImportError as e:
        print(f"[ERROR] Could not import test runner: {e}")
        print("NOTE: Make sure tests/test_runner.py exists")
        return 1
    except Exception as e:
        print(f"[ERROR] Test execution failed: {e}")
        return 1


def create_test_data_command(args) -> int:
    """Create test data fixtures using 3-tier strategy: tiny real + synthetic + dev real data."""
    try:
        import json
        from pathlib import Path

        import numpy as np
        import pandas as pd
        import scanpy as sc

        tier = getattr(args, "tier", "all")
        print(f"TESTING: Creating Test Data Fixtures - Tier: {tier}")
        print("=" * 60)
        print("3-Tier Strategy:")
        print("  PACKAGE: Tiny: Small real samples (committed to git)")
        print("  AUTO: Synthetic: Generated from metadata")
        print("  RESEARCH: Real: Full developer samples (gitignored)")
        print("")

        # For synthetic and tiny tiers, we can work from existing metadata
        if tier in ["synthetic", "tiny"]:
            print("SEARCH: Looking for existing metadata...")
            return create_from_metadata(tier, args)

        # For real tier, we need actual data files
        data_root = Path("data")
        if not data_root.exists():
            print(
                "[ERROR] Data directory not found. Place data files in data/[project]/[tissue]/ first."
            )
            print(
                "NOTE: For synthetic data: run with --tier synthetic (uses existing metadata)"
            )
            return 1

        projects_found = []
        results = []

        # Scan for project directories
        for project_dir in data_root.iterdir():
            if not project_dir.is_dir() or project_dir.name.startswith("."):
                continue

            for tissue_dir in project_dir.iterdir():
                if not tissue_dir.is_dir():
                    continue

                h5ad_files = list(tissue_dir.glob("*.h5ad"))
                if not h5ad_files:
                    continue

                # Find best source file
                original_files = [f for f in h5ad_files if "original" in f.name]
                train_files = [
                    f for f in h5ad_files if "train" in f.name and "batch" not in f.name
                ]

                if original_files:
                    data_file = original_files[0]
                elif train_files:
                    data_file = train_files[0]
                else:
                    continue

                print(f"\nFOUND: Processing: {project_dir.name}/{tissue_dir.name}")
                print(f"DATA: Source file: {data_file.name}")

                # Create test data based on tier
                if tier in ["tiny", "all"]:
                    result = create_tiny_fixtures(
                        project_dir.name, tissue_dir.name, data_file
                    )
                    if result:
                        results.append(result)

                if tier in ["synthetic", "all"]:
                    result = create_synthetic_fixtures(
                        project_dir.name, tissue_dir.name, data_file
                    )
                    if result:
                        results.append(result)

                if tier in ["real", "all"]:
                    result = create_real_fixtures(
                        project_dir.name, tissue_dir.name, data_file
                    )
                    if result:
                        results.append(result)

                projects_found.append(f"{project_dir.name}/{tissue_dir.name}")

        if not results:
            print("\n[ERROR] No test data created.")
            return 1

        # Save summary
        summary = {
            "created_at": pd.Timestamp.now().isoformat(),
            "tier": tier,
            "projects": results,
            "total_projects": len(projects_found),
        }

        summary_path = Path("tests/fixtures/test_data_summary.json")
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            summary_path.parent.mkdir(parents=True, exist_ok=True)

        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\nSUCCESS: Test data created for {len(projects_found)} project(s)")
        print(f"DOC: Summary saved to: {summary_path}")

        return 0

    except ImportError as e:
        print(f"[ERROR] Required dependencies not available: {e}")
        return 1
    except Exception as e:
        print(f"[ERROR] Failed to create test data: {e}")
        return 1


def create_from_metadata(tier, args) -> int:
    """Create test data from existing metadata files."""
    from pathlib import Path

    fixtures_root = Path("tests/fixtures")
    if not fixtures_root.exists():
        print("[ERROR] No test fixtures directory found")
        return 1

    results = []

    # Scan for existing metadata
    for project_dir in fixtures_root.iterdir():
        if not project_dir.is_dir() or project_dir.name.startswith("."):
            continue

        print(f"\nFOUND: Processing project: {project_dir.name}")

        # Look for existing metadata files
        metadata_files = list(project_dir.glob("*_stats.json"))
        for metadata_file in metadata_files:
            # Extract tissue name from filename
            # test_data_head_stats.json -> head
            tissue = metadata_file.stem.replace("test_data_", "").replace("_stats", "")

            print(f"DATA: Found metadata: {metadata_file.name} -> tissue: {tissue}")

            if tier == "tiny":
                result = create_tiny_from_metadata(
                    project_dir.name, tissue, metadata_file, args
                )
                if result:
                    results.append(result)

            elif tier == "synthetic":
                result = create_synthetic_from_metadata(
                    project_dir.name, tissue, metadata_file, args
                )
                if result:
                    results.append(result)

    if not results:
        print(f"[ERROR] No metadata found for {tier} data creation")
        return 1

    print(f"\nSUCCESS: Created {len(results)} {tier} fixtures!")
    return 0


def load_test_data_config():
    """Load test data defaults from config (dev-only, not shipped to users)."""
    from pathlib import Path

    import yaml

    config_path = Path("configs/test_data_defaults.yaml")
    if config_path.exists():
        try:
            with open(config_path) as f:
                return yaml.safe_load(f)
        except Exception:
            pass

    # Fallback defaults if config not found
    return {
        "test_data": {
            "defaults": {
                "tiny": {"cells": 50, "genes": 100},
                "synthetic": {"cells": 500, "genes": 1000},
                "real": {"cells": 5000, "genes": 2000},
            },
            "batch_correction": {"noise_std": 0.1},
            "random_seed": 42,
        }
    }


def create_tiny_from_metadata(project, tissue, metadata_file, args, seed=None):
    """Create tiny fixtures from metadata."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # Load config defaults
    config = load_test_data_config()
    defaults = config["test_data"]["defaults"]["tiny"]
    if seed is None:
        seed = config["test_data"]["random_seed"]

    np.random.seed(seed)

    try:
        print("  PACKAGE: Creating tiny fixture from metadata...")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get size parameters (allow override, fallback to config)
        n_cells = getattr(args, "cells", None) or defaults["cells"]
        n_genes = getattr(args, "genes", None) or defaults["genes"]
        batch_versions = getattr(args, "batch_versions", False)

        # Generate expression data matching real patterns
        expr_stats = metadata.get("expression_stats", {})
        synthetic_data = np.random.lognormal(
            mean=np.log(expr_stats.get("non_zero_mean", 1.0)),
            sigma=1.0,
            size=(n_cells, n_genes),
        )

        # Apply sparsity
        sparsity = expr_stats.get("sparsity", 0.8)
        mask = np.random.random((n_cells, n_genes)) < sparsity
        synthetic_data[mask] = 0

        # Create synthetic metadata matching real patterns
        obs_data = {}
        if "age_distribution" in metadata:
            ages = list(metadata["age_distribution"].keys())
            obs_data["age"] = np.random.choice(ages, n_cells)

        if "sex_distribution" in metadata:
            sexes = list(metadata["sex_distribution"].keys())
            obs_data["sex"] = np.random.choice(sexes, n_cells)

        # Add batch correction columns expected by TimeFlies
        obs_data["dataset"] = np.random.choice(["batch1", "batch2"], n_cells)
        obs_data["afca_annotation_broad"] = np.random.choice(
            ["neuron", "glia", "muscle"], n_cells
        )

        obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        # Create AnnData
        import anndata

        adata_tiny = anndata.AnnData(X=synthetic_data, obs=obs_df, var=var_df)

        # Save fixtures
        output_dir = Path("tests/fixtures") / project

        # Regular version
        tiny_path = output_dir / f"tiny_{tissue}.h5ad"
        adata_tiny.write_h5ad(tiny_path)
        files_created = [f"tiny_{tissue}.h5ad"]

        # Batch corrected version (if requested)
        if batch_versions:
            # Simulate batch correction using config parameters
            noise_std = config["test_data"]["batch_correction"]["noise_std"]
            batch_data = synthetic_data.copy()
            batch_noise = np.random.normal(0, noise_std, batch_data.shape)
            batch_data[batch_data > 0] += batch_noise[batch_data > 0]
            batch_data = np.maximum(batch_data, 0)  # Keep non-negative

            adata_batch = anndata.AnnData(X=batch_data, obs=obs_df, var=var_df)

            batch_path = output_dir / f"tiny_{tissue}_batch.h5ad"
            adata_batch.write_h5ad(batch_path)
            files_created.append(f"tiny_{tissue}_batch.h5ad")

        print(
            f"    [OK] Tiny: {', '.join(files_created)} ({n_cells} cells, {n_genes} genes)"
        )
        return {"tier": "tiny", "project": project, "tissue": tissue}

    except Exception as e:
        print(f"    [ERROR] Tiny creation failed: {e}")
        return None


def create_synthetic_from_metadata(project, tissue, metadata_file, args, seed=None):
    """Create synthetic fixtures from metadata."""
    import json
    from pathlib import Path

    import numpy as np
    import pandas as pd

    # Load config defaults
    config = load_test_data_config()
    defaults = config["test_data"]["defaults"]["synthetic"]
    if seed is None:
        seed = config["test_data"]["random_seed"]

    np.random.seed(seed)

    try:
        print("  AUTO: Creating synthetic fixture from metadata...")

        with open(metadata_file) as f:
            metadata = json.load(f)

        # Get size parameters (allow override, fallback to config)
        n_cells = getattr(args, "cells", None) or defaults["cells"]
        n_genes = getattr(args, "genes", None) or defaults["genes"]
        batch_versions = getattr(args, "batch_versions", False)

        # Generate expression data matching real patterns
        expr_stats = metadata.get("expression_stats", {})
        synthetic_data = np.random.lognormal(
            mean=np.log(expr_stats.get("non_zero_mean", 1.0)),
            sigma=1.0,
            size=(n_cells, n_genes),
        )

        # Apply sparsity
        sparsity = expr_stats.get("sparsity", 0.8)
        mask = np.random.random((n_cells, n_genes)) < sparsity
        synthetic_data[mask] = 0

        # Create synthetic metadata
        obs_data = {}
        if "age_distribution" in metadata:
            ages = list(metadata["age_distribution"].keys())
            obs_data["age"] = np.random.choice(ages, n_cells)

        if "sex_distribution" in metadata:
            sexes = list(metadata["sex_distribution"].keys())
            obs_data["sex"] = np.random.choice(sexes, n_cells)

        # Add batch correction columns expected by TimeFlies
        obs_data["dataset"] = np.random.choice(["batch1", "batch2"], n_cells)
        obs_data["afca_annotation_broad"] = np.random.choice(
            ["neuron", "glia", "muscle"], n_cells
        )

        obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
        var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

        # Create AnnData
        import anndata

        adata_synthetic = anndata.AnnData(X=synthetic_data, obs=obs_df, var=var_df)

        # Save fixtures
        output_dir = Path("tests/fixtures") / project

        # Regular version
        synthetic_path = output_dir / f"synthetic_{tissue}.h5ad"
        adata_synthetic.write_h5ad(synthetic_path)
        files_created = [f"synthetic_{tissue}.h5ad"]

        # Batch corrected version (if requested)
        if batch_versions:
            # Simulate batch correction using config parameters
            noise_std = config["test_data"]["batch_correction"]["noise_std"]
            batch_data = synthetic_data.copy()
            batch_noise = np.random.normal(0, noise_std, batch_data.shape)
            batch_data[batch_data > 0] += batch_noise[batch_data > 0]
            batch_data = np.maximum(batch_data, 0)

            adata_batch = anndata.AnnData(X=batch_data, obs=obs_df, var=var_df)

            batch_path = output_dir / f"synthetic_{tissue}_batch.h5ad"
            adata_batch.write_h5ad(batch_path)
            files_created.append(f"synthetic_{tissue}_batch.h5ad")

        print(
            f"    [OK] Synthetic: {', '.join(files_created)} ({n_cells} cells, {n_genes} genes)"
        )
        return {"tier": "synthetic", "project": project, "tissue": tissue}

    except Exception as e:
        print(f"    [ERROR] Synthetic creation failed: {e}")
        return None


def create_tiny_fixtures(project, tissue, data_file, seed=42):
    """Create tiny real data fixtures (committed to git) - Tier 1."""
    import json
    from pathlib import Path

    import numpy as np
    import scanpy as sc

    np.random.seed(seed)

    try:
        print("  PACKAGE: Creating tiny fixtures...")
        adata = sc.read_h5ad(data_file)

        # Very small samples - suitable for git
        n_cells = min(50, adata.n_obs)
        n_genes = min(100, adata.n_vars)

        cell_indices = np.random.choice(adata.n_obs, n_cells, replace=False)
        gene_indices = np.random.choice(adata.n_vars, n_genes, replace=False)

        adata_tiny = adata[cell_indices, gene_indices].copy()

        # Save to fixtures directory
        output_dir = Path("tests/fixtures") / project
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            output_dir.mkdir(parents=True, exist_ok=True)

        tiny_path = output_dir / f"tiny_{tissue}.h5ad"
        adata_tiny.write_h5ad(tiny_path)

        # Generate and save metadata
        stats = generate_data_stats(adata_tiny, project, tissue, tier="tiny")
        metadata_path = output_dir / f"tiny_{tissue}_metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        print(f"    [OK] Tiny: {tiny_path} ({n_cells} cells, {n_genes} genes)")

        return {"tier": "tiny", "project": project, "tissue": tissue, "stats": stats}

    except Exception as e:
        print(f"    [ERROR] Tiny fixtures failed: {e}")
        return None


def create_synthetic_fixtures(project, tissue, data_file, seed=42):
    """Create synthetic data from metadata - Tier 2."""
    import json
    from pathlib import Path

    import numpy as np

    np.random.seed(seed)

    try:
        print("  AUTO: Creating synthetic fixtures...")

        # First check if we have existing metadata to use
        output_dir = Path("tests/fixtures") / project
        metadata_path = output_dir / f"tiny_{tissue}_metadata.json"

        if metadata_path.exists():
            # Use existing metadata to generate synthetic data
            with open(metadata_path) as f:
                metadata = json.load(f)

            # Generate synthetic data matching the patterns
            n_cells = 500  # Medium size
            n_genes = 1000

            # Create synthetic expression matrix based on real stats
            expr_stats = metadata.get("expression_stats", {})

            # Generate data matching real distributions
            synthetic_data = np.random.lognormal(
                mean=np.log(expr_stats.get("non_zero_mean", 1.0)),
                sigma=1.0,
                size=(n_cells, n_genes),
            )

            # Apply sparsity pattern
            sparsity = expr_stats.get("sparsity", 0.8)
            mask = np.random.random((n_cells, n_genes)) < sparsity
            synthetic_data[mask] = 0

            # Create synthetic AnnData object
            import pandas as pd

            # Create synthetic metadata matching real patterns
            obs_data = {}
            if "age_distribution" in metadata:
                ages = list(metadata["age_distribution"].keys())
                obs_data["age"] = np.random.choice(ages, n_cells)

            if "sex_distribution" in metadata:
                sexes = list(metadata["sex_distribution"].keys())
                obs_data["sex"] = np.random.choice(sexes, n_cells)

            # Add batch correction columns expected by TimeFlies
            obs_data["dataset"] = np.random.choice(["batch1", "batch2"], n_cells)
            obs_data["afca_annotation_broad"] = np.random.choice(
                ["neuron", "glia", "muscle"], n_cells
            )

            obs_df = pd.DataFrame(obs_data, index=[f"cell_{i}" for i in range(n_cells)])
            var_df = pd.DataFrame(index=[f"gene_{i}" for i in range(n_genes)])

            # Create AnnData
            import anndata

            adata_synthetic = anndata.AnnData(X=synthetic_data, obs=obs_df, var=var_df)

            # Save synthetic data
            synthetic_path = output_dir / f"synthetic_{tissue}.h5ad"
            adata_synthetic.write_h5ad(synthetic_path)

            print(
                f"    [OK] Synthetic: {synthetic_path} ({n_cells} cells, {n_genes} genes)"
            )

            return {
                "tier": "synthetic",
                "project": project,
                "tissue": tissue,
                "size": (n_cells, n_genes),
            }

        else:
            print("    WARNING:  No metadata found for synthetic generation")
            return None

    except Exception as e:
        print(f"    [ERROR] Synthetic fixtures failed: {e}")
        return None


def create_real_fixtures(project, tissue, data_file, seed=42):
    """Create full-scale real data fixtures (gitignored) - Tier 3."""
    from pathlib import Path

    import numpy as np
    import scanpy as sc

    np.random.seed(seed)

    try:
        print("  RESEARCH: Creating real fixtures...")
        adata = sc.read_h5ad(data_file)

        # Larger realistic samples for thorough testing
        n_cells = min(5000, adata.n_obs)
        n_genes = min(2000, adata.n_vars)

        cell_indices = np.random.choice(adata.n_obs, n_cells, replace=False)
        gene_indices = np.random.choice(adata.n_vars, n_genes, replace=False)

        adata_real = adata[cell_indices, gene_indices].copy()

        # Save to fixtures directory (will be gitignored)
        output_dir = Path("tests/fixtures") / project
        # Create directory if it doesn't exist (skip during tests)
        if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
            output_dir.mkdir(parents=True, exist_ok=True)

        real_path = output_dir / f"real_{tissue}.h5ad"
        adata_real.write_h5ad(real_path)

        print(f"    [OK] Real: {real_path} ({n_cells} cells, {n_genes} genes)")

        return {
            "tier": "real",
            "project": project,
            "tissue": tissue,
            "size": (n_cells, n_genes),
        }

    except Exception as e:
        print(f"    [ERROR] Real fixtures failed: {e}")
        return None


def generate_data_stats(adata, project, tissue, batch_corrected=False, tier=None):
    """Generate comprehensive statistics for test data."""
    import numpy as np
    import pandas as pd

    # Basic info
    stats = {
        "project": project,
        "tissue": tissue,
        "tier": tier,
        "n_obs": int(adata.n_obs),
        "n_vars": int(adata.n_vars),
        "batch_corrected": batch_corrected,
        "created_at": pd.Timestamp.now().isoformat(),
    }

    # Metadata distributions if available
    if "age" in adata.obs.columns:
        stats["age_distribution"] = adata.obs["age"].value_counts().to_dict()

    if "sex" in adata.obs.columns:
        stats["sex_distribution"] = adata.obs["sex"].value_counts().to_dict()

    # Cell type distribution (try different column names)
    cell_type_cols = ["afca_annotation_broad", "cell_type", "celltype", "annotation"]
    for col in cell_type_cols:
        if col in adata.obs.columns:
            stats["cell_types"] = adata.obs[col].value_counts().head(10).to_dict()
            break

    # Expression data statistics
    if hasattr(adata.X, "toarray"):
        X_array = adata.X.toarray()
    else:
        X_array = adata.X

    stats["expression_stats"] = {
        "min": float(np.min(X_array)),
        "max": float(np.max(X_array)),
        "mean": float(np.mean(X_array)),
        "median": float(np.median(X_array)),
        "sparsity": float(np.mean(X_array == 0)),
        "non_zero_mean": float(np.mean(X_array[X_array > 0]))
        if np.any(X_array > 0)
        else 0.0,
    }

    # Gene info if available
    if hasattr(adata.var, "columns"):
        stats["gene_info"] = {
            "columns": list(adata.var.columns),
            "n_highly_variable": int(
                adata.var.get(
                    "highly_variable", pd.Series([False] * adata.n_vars)
                ).sum()
            ),
        }

    return stats
