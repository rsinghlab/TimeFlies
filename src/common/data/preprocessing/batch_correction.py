"""
TimeFlies Batch Correction using scVI.

Implements proper ML workflow for batch correction:
1. Train scVI model on train split only
2. Use prepare_query_anndata for eval split (prevents data leakage)
3. Transform both splits with trained model
4. Save results with batch-corrected data
"""

import os
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

# Defer dependency check until class instantiation
BATCH_CORRECTION_AVAILABLE = True  # Will be checked in __init__
scvi = None
torch = None
sm = None


class BatchCorrector:
    """
    TimeFlies batch correction using scVI with proper train/eval workflow.

    This class implements the correct ML process:
    1. Train scVI model on train data only
    2. Use prepare_query_anndata for eval data (prevents data leakage)
    3. Transform both train and eval with trained model
    4. Save results with original counts preserved in layers['counts']
    """

    def __init__(
        self,
        tissue: str,
        base_dir: Path,
        project: str,
    ):
        """
        Initialize BatchCorrector with TimeFlies integration.

        Args:
            tissue: tissue type (e.g. 'head', 'body')
            base_dir: base directory containing data files
            project: project name for overrides (e.g. 'fruitfly_aging')
        """
        print(f"DEBUG: BatchCorrector init start - tissue={tissue}, project={project}")

        # Check batch correction dependencies
        try:
            global scvi, torch, sm
            import scib.metrics as sm
            import scvi
            import torch

            print("DEBUG: Dependencies imported successfully")
        except ImportError as e:
            print(f"DEBUG: Dependency import failed: {e}")
            raise ImportError(
                "Batch correction dependencies not available. "
                "Install with: pip install scvi-tools scib"
            )

        self.tissue = tissue
        self.base_dir = Path(base_dir)
        self.project = project
        print("DEBUG: Basic properties set")

        # Load batch correction configuration
        print("DEBUG: Loading batch config...")
        self._load_batch_config()
        print("DEBUG: Batch config loaded")

        # Load configuration from batch_correction.yaml
        print("DEBUG: Setting up config...")
        self._setup_config()
        print("DEBUG: Config setup complete")

        # Set up PyTorch and scVI settings
        print("DEBUG: Setting up PyTorch...")
        self._setup_pytorch()
        print("DEBUG: PyTorch setup complete")

    def _load_batch_config(self):
        """Load batch correction configuration from batch_correction.yaml."""
        from pathlib import Path as ConfigPath

        import yaml

        # Find batch_correction.yaml config file
        config_paths = [
            ConfigPath("configs/batch_correction.yaml"),  # Current directory
            ConfigPath(__file__).parent.parent.parent.parent
            / "configs/batch_correction.yaml",  # Repo structure
            ConfigPath.home()
            / ".timeflies/configs/batch_correction.yaml",  # User config
        ]

        config_path = None
        for path in config_paths:
            if path.exists():
                config_path = path
                break

        if not config_path:
            raise FileNotFoundError(
                "batch_correction.yaml not found. Please ensure it exists in configs/ directory."
            )

        # Load the YAML config
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)

        # Apply project-specific overrides if available
        if (
            "project_overrides" in config_dict
            and self.project in config_dict["project_overrides"]
        ):
            overrides = config_dict["project_overrides"][self.project]
            # Recursively merge overrides into the main config
            config_dict = self._merge_configs(config_dict, overrides)

        # Convert to a simple object for attribute access
        class Config:
            def __init__(self, d):
                for k, v in d.items():
                    if isinstance(v, dict):
                        setattr(self, k, Config(v))
                    else:
                        setattr(self, k, v)

        self.config = Config(config_dict)

    def _merge_configs(self, base_dict: dict, override_dict: dict) -> dict:
        """Recursively merge override config into base config."""
        result = base_dict.copy()

        for key, value in override_dict.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override or add new key
                result[key] = value

        return result

    def _setup_config(self):
        """Setup configuration from batch_correction.yaml integrated with TimeFlies."""
        batch_config = self.config.batch_correction

        # Data processing settings
        self.batch_column = batch_config.preprocessing.batch_column
        self.label_column = batch_config.preprocessing.label_column
        self.library_size = batch_config.preprocessing.library_size

        # Output settings
        self.scvi_latent_key = batch_config.output.scvi_latent_key
        self.scvi_normalized_key = batch_config.output.scvi_normalized_key

        # Model parameters
        model_config = batch_config.model
        self.n_latent = model_config.n_latent
        self.n_hidden = model_config.n_hidden
        self.n_layers = model_config.n_layers
        self.dropout_rate = model_config.dropout_rate

        # Training parameters
        training_config = batch_config.training
        self.max_epochs = training_config.max_epochs
        self.weight_decay = training_config.weight_decay

        # Evaluation settings
        eval_config = batch_config.evaluation
        self.run_metrics = eval_config.run_metrics
        self.run_visualization = eval_config.run_visualization
        self.plot_columns = eval_config.plot_columns

    def _setup_pytorch(self):
        """Setup PyTorch and scVI settings from configuration."""
        pytorch_config = self.config.pytorch

        torch.set_float32_matmul_precision(pytorch_config.float32_matmul_precision)
        scvi.settings.dl_num_workers = (
            os.cpu_count() - 1
            if pytorch_config.dl_num_workers == -1
            else pytorch_config.dl_num_workers
        )
        scvi.settings.dl_pin_memory = pytorch_config.dl_pin_memory
        scvi.settings.seed = pytorch_config.seed

        print(f"Using scvi-tools version: {scvi.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU for training")

    def _find_split_files(self) -> tuple[Path, Path]:
        """Find train and eval split files."""
        # Look for train file
        train_pattern = f"*_{self.tissue}_*_train.h5ad"
        train_matches = list(self.base_dir.rglob(train_pattern))
        if not train_matches:
            train_pattern = "*_train.h5ad"
            train_matches = list(self.base_dir.rglob(train_pattern))

        # Look for eval file
        eval_pattern = f"*_{self.tissue}_*_eval.h5ad"
        eval_matches = list(self.base_dir.rglob(eval_pattern))
        if not eval_matches:
            eval_pattern = "*_eval.h5ad"
            eval_matches = list(self.base_dir.rglob(eval_pattern))

        if not train_matches or not eval_matches:
            raise FileNotFoundError(
                "Could not find train/eval split files. "
                "Run 'python run_timeflies.py split' first."
            )

        return train_matches[0], eval_matches[0]

    def _find_batch_split_files(self) -> tuple[Path, Path]:
        """Find existing batch-corrected train and eval files."""
        train_pattern = f"*_{self.tissue}_*_train_batch.h5ad"
        train_matches = list(self.base_dir.rglob(train_pattern))
        if not train_matches:
            train_pattern = "*_train_batch.h5ad"
            train_matches = list(self.base_dir.rglob(train_pattern))

        eval_pattern = f"*_{self.tissue}_*_eval_batch.h5ad"
        eval_matches = list(self.base_dir.rglob(eval_pattern))
        if not eval_matches:
            eval_pattern = "*_eval_batch.h5ad"
            eval_matches = list(self.base_dir.rglob(eval_pattern))

        if not train_matches or not eval_matches:
            raise FileNotFoundError(
                "Could not find batch-corrected train/eval files. "
                "Run batch correction first."
            )

        return train_matches[0], eval_matches[0]

    def _prep_counts(self, adata):
        """Copy raw counts into .layers['counts'] for scVI."""
        adata.layers["counts"] = adata.X.copy()

    def preprocess_data(self, adata_train, adata_eval):
        """Prepare data for scVI training by adding counts layer."""
        print("Preparing data for scVI...")
        self._prep_counts(adata_train)
        self._prep_counts(adata_eval)
        print("Data preparation complete.")

    def setup_scvi(self, adata_train):
        """Setup AnnData for scVI model using train data."""
        print("Setting up scVI model...")
        scvi.model.SCVI.setup_anndata(
            adata_train,
            layer="counts",
            batch_key=self.batch_column,
            labels_key=(
                self.label_column
                if self.label_column in adata_train.obs.columns
                else None
            ),
        )
        print("scVI setup complete.")

    def train_model(self, adata_train):
        """Train the scVI model on train data only."""
        print("Training scVI model...")
        model = scvi.model.SCVI(
            adata_train,
            n_latent=self.n_latent,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            dropout_rate=self.dropout_rate,
        )

        # Train with early stopping
        model.train(
            max_epochs=self.max_epochs, plan_kwargs={"weight_decay": self.weight_decay}
        )
        print("scVI model training complete.")
        return model

    def add_scvi_outputs(self, model, adata_train, adata_eval):
        """Add scVI latent representations and normalized expression."""
        print("Generating scVI outputs...")

        # Process train data
        adata_train.obsm[self.scvi_latent_key] = model.get_latent_representation(
            adata_train
        )
        adata_train.layers[self.scvi_normalized_key] = model.get_normalized_expression(
            adata_train, library_size=self.library_size
        )

        # Process eval data (CRITICAL: use prepare_query_anndata for no data leakage)
        print("Preparing eval data as query (prevents data leakage)...")
        scvi.model.SCVI.prepare_query_anndata(adata_eval, reference_model=model)

        # Get latent representation for eval data
        adata_eval.obsm[self.scvi_latent_key] = model.get_latent_representation(
            adata_eval
        )
        adata_eval.layers[self.scvi_normalized_key] = model.get_normalized_expression(
            adata_eval, library_size=self.library_size
        )

        print("scVI outputs generated.")

    def save_results(self, adata_train, adata_eval):
        """Save batch-corrected results."""
        output_dir = self.base_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate output filenames based on input patterns
        train_stem = self._generate_output_filename(adata_train, "train")
        eval_stem = self._generate_output_filename(adata_eval, "eval")

        train_output = output_dir / f"{train_stem}_batch.h5ad"
        eval_output = output_dir / f"{eval_stem}_batch.h5ad"

        adata_train.write_h5ad(train_output)
        adata_eval.write_h5ad(eval_output)

        print(f"Saved batch-corrected train to: {train_output}")
        print(f"Saved batch-corrected eval to: {eval_output}")

    def _generate_output_filename(self, adata, split_type: str) -> str:
        """Generate consistent output filename."""
        # Try to extract project name from various sources
        project = adata.uns.get("project") or adata.uns.get("dataset") or "project"

        # Create filename pattern matching TimeFlies convention
        return f"{project}_{self.tissue}_{split_type}"

    def run_batch_correction(self):
        """Run the complete batch correction workflow."""
        start_time = time.time()

        print("🧬 Starting TimeFlies batch correction...")
        print("=" * 60)

        # Load data splits
        train_file, eval_file = self._find_split_files()
        adata_train = sc.read_h5ad(train_file)
        adata_eval = sc.read_h5ad(eval_file)
        print(f"Loaded train: {adata_train.shape}, eval: {adata_eval.shape}")

        # Run batch correction workflow
        self.preprocess_data(adata_train, adata_eval)
        self.setup_scvi(adata_train)
        model = self.train_model(adata_train)
        self.add_scvi_outputs(model, adata_train, adata_eval)
        self.save_results(adata_train, adata_eval)

        elapsed = time.time() - start_time
        print("=" * 60)
        print(f"✅ Batch correction completed in {elapsed:.1f}s")

        # Run optional evaluations if configured
        if self.run_visualization:
            self.generate_umap_visualizations()
        if self.run_metrics:
            self.evaluate_batch_correction_metrics()

        return model

    def generate_umap_visualizations(self):
        """Generate UMAP visualizations comparing uncorrected vs batch-corrected."""
        start_time = time.time()

        print("🎨 Generating UMAP visualizations...")

        # Get output directory from config
        project_name = getattr(self.config, "project", "project")
        output_dir = Path(
            self.config.batch_correction.evaluation.umap_output_dir.format(
                project=project_name
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        # Set scanpy figure directory
        sc.settings.figdir = str(output_dir)

        self._visualize_splits()

        elapsed = time.time() - start_time
        print(f"✅ UMAP visualizations completed in {elapsed:.1f}s")
        print(f"   Saved to: {output_dir}")

    def _visualize_splits(self):
        """Generate UMAPs for train/eval splits using eval data."""
        print("Generating UMAPs for eval data...")

        # Load uncorrected eval data
        _, eval_file = self._find_split_files()
        adata_uncorrected = sc.read_h5ad(eval_file)

        # Load batch-corrected eval data
        _, eval_batch_file = self._find_batch_split_files()
        adata_corrected = sc.read_h5ad(eval_batch_file)

        self._create_umap_plots(adata_uncorrected, adata_corrected, "eval")

    def _create_umap_plots(self, adata_uncorrected, adata_corrected, prefix: str):
        """Create UMAP plots comparing uncorrected vs corrected."""

        # Prepare uncorrected data for UMAP
        adata_uncorrected.layers["counts"] = adata_uncorrected.X.copy()
        sc.pp.normalize_total(adata_uncorrected, target_sum=1e4)
        sc.pp.log1p(adata_uncorrected)
        sc.pp.pca(adata_uncorrected, n_comps=30)
        sc.pp.neighbors(adata_uncorrected, n_pcs=30)
        sc.tl.umap(adata_uncorrected, min_dist=0.3)

        # Prepare corrected data for UMAP (use scVI latent space)
        sc.pp.neighbors(adata_corrected, use_rep=self.scvi_latent_key, n_neighbors=20)
        sc.tl.umap(adata_corrected, min_dist=0.3)

        # Columns to plot (from configuration)
        plot_cols = []
        for col in self.plot_columns:
            if col in adata_corrected.obs.columns:
                plot_cols.append(col)

        # Convert numeric columns to string for plotting
        for col in plot_cols:
            for adata in [adata_uncorrected, adata_corrected]:
                if col in adata.obs.columns and adata.obs[col].dtype.kind in "ifc":
                    adata.obs[f"{col}_str"] = adata.obs[col].astype(str)

        # Generate individual plots
        for col in plot_cols:
            plot_col = (
                f"{col}_str" if f"{col}_str" in adata_uncorrected.obs.columns else col
            )

            # Uncorrected plot
            sc.pl.umap(
                adata_uncorrected,
                color=plot_col,
                title=f"Uncorrected: {col}",
                save=f"_{prefix}_uncorrected_{col}.png",
                show=False,
            )
            plt.close()

            # Batch-corrected plot
            sc.pl.umap(
                adata_corrected,
                color=plot_col,
                title=f"Batch-corrected: {col}",
                save=f"_{prefix}_scvi_{col}.png",
                show=False,
            )
            plt.close()

        # Side-by-side comparison plots
        if plot_cols:
            plot_cols_str = [
                f"{col}_str" if f"{col}_str" in adata_corrected.obs.columns else col
                for col in plot_cols
            ]

            # Uncorrected side-by-side
            sc.pl.umap(
                adata_uncorrected,
                color=plot_cols_str,
                frameon=False,
                ncols=min(3, len(plot_cols_str)),
                title=[f"Uncorrected: {col}" for col in plot_cols],
                save=f"_{prefix}_uncorrected_all.png",
                show=False,
            )
            plt.close()

            # Batch-corrected side-by-side
            sc.pl.umap(
                adata_corrected,
                color=plot_cols_str,
                frameon=False,
                ncols=min(3, len(plot_cols_str)),
                title=[f"Batch-corrected: {col}" for col in plot_cols],
                save=f"_{prefix}_scvi_all.png",
                show=False,
            )
            plt.close()

    def evaluate_batch_correction_metrics(self):
        """Evaluate batch correction quality using scIB metrics."""
        start_time = time.time()

        print("📊 Evaluating batch correction metrics...")

        # Get output directory from config
        project_name = getattr(self.config, "project", "project")
        output_dir = Path(
            self.config.batch_correction.evaluation.metrics_output_dir.format(
                project=project_name
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        results_df = self._evaluate_splits()
        output_file = output_dir / "eval_batch_metrics.csv"

        # Save results
        results_df.to_csv(output_file, index=False)

        elapsed = time.time() - start_time
        print("📋 Batch correction metrics:")
        print(results_df.to_string(index=False))
        print(f"✅ Metrics evaluation completed in {elapsed:.1f}s")
        print(f"   Saved to: {output_file}")

        return results_df

    def _evaluate_splits(self) -> pd.DataFrame:
        """Evaluate batch correction on eval split."""
        # Load uncorrected and corrected eval data
        _, eval_file = self._find_split_files()
        adata_uncorrected = sc.read_h5ad(eval_file)

        _, eval_batch_file = self._find_batch_split_files()
        adata_corrected = sc.read_h5ad(eval_batch_file)

        return self._compute_metrics(adata_uncorrected, adata_corrected)

    def _compute_metrics(self, adata_uncorrected, adata_corrected) -> pd.DataFrame:
        """Compute scIB and silhouette metrics."""

        # Prepare data for scIB metrics
        adata_uncorrected.layers["counts"] = adata_uncorrected.X.copy()
        adata_corrected.obsm["X_int"] = adata_corrected.obsm[self.scvi_latent_key]

        # Compute neighbors on integrated embedding
        sc.pp.neighbors(adata_corrected, use_rep="X_int", n_neighbors=15)

        # Compute scIB metrics
        try:
            scores_wide = sm.metrics_fast(
                adata_uncorrected,
                adata_corrected,
                batch_key=self.batch_column,
                label_key=(
                    self.label_column
                    if self.label_column in adata_corrected.obs.columns
                    else None
                ),
                embed="X_int",
                type_="embed",
                n_cores=max(1, os.cpu_count() - 1),
            )

            # Extract metrics
            metrics_list = []
            scores_list = []

            for metric_name in scores_wide.index:
                score_value = scores_wide.iloc[
                    scores_wide.index.get_loc(metric_name), 0
                ]
                if pd.notna(score_value):
                    metrics_list.append(str(metric_name))
                    scores_list.append(float(score_value))

        except Exception as e:
            print(f"Warning: scIB metrics failed: {e}")
            metrics_list = []
            scores_list = []

        # Add silhouette scores for available columns (from config)
        for col in self.plot_columns:
            if col in adata_corrected.obs.columns:
                try:
                    # Convert to string if numeric
                    if adata_corrected.obs[col].dtype.kind in "ifc":
                        adata_corrected.obs[f"{col}_str"] = adata_corrected.obs[
                            col
                        ].astype(str)
                        col = f"{col}_str"

                    sil_score = sm.silhouette(
                        adata_corrected, label_key=col, embed="X_int"
                    )
                    metrics_list.append(f"silhouette_{col.replace('_str', '')}")
                    scores_list.append(sil_score)
                except Exception as e:
                    print(f"Warning: Silhouette score for {col} failed: {e}")

        # Create results DataFrame
        if metrics_list:
            results_df = pd.DataFrame({"metric": metrics_list, "score": scores_list})
        else:
            results_df = pd.DataFrame({"metric": ["no_metrics"], "score": [0.0]})

        return results_df
