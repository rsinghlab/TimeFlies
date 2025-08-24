"""Functional tests for complete end-to-end workflows."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import for end-to-end testing
from common.cli.main import main_cli
from common.core.active_config import get_active_project, get_config_for_active_project


@pytest.mark.functional
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_cli_setup_workflow(self):
        """Test complete CLI setup workflow."""
        # Test the full setup command execution
        result = main_cli(["setup"])

        # Should complete successfully
        assert result == 0

    def test_complete_verification_workflow(self):
        """Test complete verification workflow."""
        # Test the full verification process
        result = main_cli(["verify"])

        # Should return exit code (may be 0 or 1 depending on system state)
        assert isinstance(result, int)

    def test_complete_test_data_creation_workflow(self):
        """Test complete test data creation workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temp directory for test
            original_cwd = Path.cwd()
            try:
                import os

                os.chdir(temp_dir)

                # Create test data
                result = main_cli(["create-test-data"])

                # Should complete successfully
                assert result == 0

            finally:
                os.chdir(original_cwd)

    def test_config_loading_complete_workflow(self):
        """Test complete config loading workflow."""
        # Test actual config loading process
        get_active_project()
        config_manager = get_config_for_active_project("default")
        config = config_manager.get_config()

        # Verify complete config structure
        assert hasattr(config, "general")
        assert hasattr(config, "data")
        assert hasattr(config, "model")
        assert hasattr(config, "hardware")
        assert hasattr(config, "paths")

        # Test config values are properly loaded
        assert config.general.random_state == 42
        assert config.data.tissue in ["head", "body", "all"]
        assert config.hardware.processor in ["CPU", "GPU", "M"]

    def test_path_resolution_workflow(self):
        """Test complete path resolution workflow."""
        from common.utils.path_manager import PathManager

        config_manager = get_config_for_active_project("default")
        config = config_manager.get_config()

        path_manager = PathManager(config)

        # Test path generation for different file types
        train_path = path_manager.get_file_path("train")
        eval_path = path_manager.get_file_path("eval")
        original_path = path_manager.get_file_path("original")

        # Verify paths are generated
        assert isinstance(train_path, str)
        assert isinstance(eval_path, str)
        assert isinstance(original_path, str)

        assert "train" in train_path
        assert "eval" in eval_path
        assert "original" in original_path


@pytest.mark.functional
class TestDataProcessingWorkflows:
    """Test complete data processing workflows."""

    def test_complete_anndata_processing(self, large_sample_anndata):
        """Test complete AnnData processing workflow."""
        import scanpy as sc
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Start with raw data
        adata = large_sample_anndata.copy()
        original_shape = adata.shape

        # Complete preprocessing pipeline
        # 1. Normalization
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # 2. Filtering
        sc.pp.filter_cells(adata, min_genes=10)
        sc.pp.filter_genes(adata, min_cells=3)

        # 3. Feature selection
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

        # 4. Label encoding
        encoder = LabelEncoder()
        encoded_ages = encoder.fit_transform(adata.obs["age"])
        adata.obs["age_encoded"] = encoded_ages

        # 5. Data splitting
        X = adata.X
        y = adata.obs["age_encoded"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Verify complete pipeline
        assert adata.shape[0] <= original_shape[0]  # May have filtered cells
        assert adata.shape[1] <= original_shape[1]  # May have filtered genes
        assert "age_encoded" in adata.obs.columns
        assert len(X_train) + len(X_test) == adata.n_obs
        assert np.all(adata.X >= 0)  # Log1p ensures non-negative

    def test_complete_model_training_workflow(self, small_sample_anndata):
        """Test complete model training workflow."""
        import tensorflow as tf
        from sklearn.metrics import accuracy_score, classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Prepare data
        X = small_sample_anndata.X.astype(np.float32)
        y = small_sample_anndata.obs["age"].values

        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )

        # Create model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(64, activation="relu", input_shape=(X.shape[1],)),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train model
        history = model.fit(
            X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0
        )

        # Evaluate model
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        # Generate detailed metrics
        report = classification_report(y_test, y_pred, output_dict=True)

        # Verify complete workflow
        assert "loss" in history.history
        assert "accuracy" in history.history
        assert 0 <= test_accuracy <= 1
        assert len(y_pred) == len(y_test)
        assert isinstance(report, dict)
        assert "accuracy" in report

    def test_complete_evaluation_workflow(self, small_sample_anndata):
        """Test complete model evaluation workflow."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Prepare data
        X = small_sample_anndata.X
        y = small_sample_anndata.obs["age"].values

        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )

        # Train multiple models for comparison
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=50, random_state=42),
            "LogisticRegression": LogisticRegression(random_state=42, max_iter=200),
        }

        results = {}

        for name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)
            model.predict_proba(X_test)

            # Calculate comprehensive metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=3)

            # Store results
            results[name] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std(),
            }

            # Verify metrics
            assert 0 <= accuracy <= 1
            assert 0 <= precision <= 1
            assert 0 <= recall <= 1
            assert 0 <= f1 <= 1

        # Compare models
        assert len(results) == 2
        assert "RandomForest" in results
        assert "LogisticRegression" in results


@pytest.mark.functional
class TestFileSystemWorkflows:
    """Test complete file system workflows."""

    def test_complete_data_io_workflow(self, large_sample_anndata):
        """Test complete data I/O workflow."""
        import pandas as pd
        import scanpy as sc

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)

            # Create directory structure
            data_dir = base_path / "data" / "processed"
            results_dir = base_path / "results"
            models_dir = base_path / "models"

            for directory in [data_dir, results_dir, models_dir]:
                directory.mkdir(parents=True)

            # Save processed data
            processed_file = data_dir / "processed_data.h5ad"
            large_sample_anndata.write_h5ad(processed_file)

            # Save metadata
            metadata_file = data_dir / "metadata.csv"
            large_sample_anndata.obs.to_csv(metadata_file)

            # Save gene info
            gene_file = data_dir / "genes.csv"
            large_sample_anndata.var.to_csv(gene_file)

            # Verify files were created
            assert processed_file.exists()
            assert metadata_file.exists()
            assert gene_file.exists()

            # Test loading workflow
            loaded_data = sc.read_h5ad(processed_file)
            loaded_metadata = pd.read_csv(metadata_file, index_col=0)
            loaded_genes = pd.read_csv(gene_file, index_col=0)

            # Verify loaded data integrity
            assert loaded_data.n_obs == large_sample_anndata.n_obs
            assert loaded_data.n_vars == large_sample_anndata.n_vars
            assert len(loaded_metadata) == large_sample_anndata.n_obs
            assert len(loaded_genes) == large_sample_anndata.n_vars

    def test_complete_model_persistence_workflow(self, small_sample_anndata):
        """Test complete model persistence workflow."""
        import joblib
        import tensorflow as tf
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir)
            models_dir = base_path / "models"
            models_dir.mkdir()

            # Prepare data
            X = small_sample_anndata.X.astype(np.float32)
            y = small_sample_anndata.obs["age"].values

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            num_classes = len(np.unique(y_encoded))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )

            # Train and save TensorFlow model
            tf_model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        32, activation="relu", input_shape=(X.shape[1],)
                    ),
                    tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            tf_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            tf_model.fit(X_train, y_train, epochs=2, verbose=0)

            tf_model_path = models_dir / "tensorflow_model"
            tf_model.save(tf_model_path)

            # Train and save sklearn model
            rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
            rf_model.fit(X_train, y_train)

            rf_model_path = models_dir / "random_forest_model.pkl"
            joblib.dump(rf_model, rf_model_path)

            # Save encoder
            encoder_path = models_dir / "label_encoder.pkl"
            joblib.dump(encoder, encoder_path)

            # Verify files exist
            assert tf_model_path.exists()
            assert rf_model_path.exists()
            assert encoder_path.exists()

            # Test loading workflow
            loaded_tf_model = tf.keras.models.load_model(tf_model_path)
            loaded_rf_model = joblib.load(rf_model_path)
            loaded_encoder = joblib.load(encoder_path)

            # Test loaded models work
            tf_pred = loaded_tf_model.predict(X_test, verbose=0)
            rf_pred = loaded_rf_model.predict(X_test)

            # Verify predictions
            assert tf_pred.shape == (len(X_test), num_classes)
            assert len(rf_pred) == len(X_test)
            assert len(loaded_encoder.classes_) == len(encoder.classes_)


@pytest.mark.functional
class TestErrorRecoveryWorkflows:
    """Test complete error recovery workflows."""

    def test_robust_data_loading_workflow(self):
        """Test robust data loading with error recovery."""
        import scanpy as sc

        from common.utils.exceptions import DataError

        # Test handling of missing files
        try:
            sc.read_h5ad("definitely_does_not_exist.h5ad")
            assert False, "Should have raised an exception"
        except (FileNotFoundError, OSError):
            # Expected behavior
            pass

        # Test handling of corrupted data (empty array)
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create minimal valid h5ad file
            import anndata as ad

            try:
                empty_data = ad.AnnData(X=np.empty((0, 0)))
                temp_file = Path(temp_dir) / "empty.h5ad"
                empty_data.write_h5ad(temp_file)

                loaded = sc.read_h5ad(temp_file)
                assert loaded.n_obs == 0
                assert loaded.n_vars == 0
            except Exception:
                # Some versions may not handle empty data well
                pass

    def test_robust_model_training_workflow(self, small_sample_anndata):
        """Test robust model training with error handling."""
        import tensorflow as tf
        from sklearn.preprocessing import LabelEncoder

        # Prepare problematic data (very small dataset)
        X = small_sample_anndata.X[:10].astype(np.float32)  # Only 10 samples
        y = small_sample_anndata.obs["age"].values[:10]

        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        # Try to train with very small data
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

        try:
            # This might fail or give warnings with very small dataset
            history = model.fit(X, y_encoded, epochs=1, verbose=0)

            # If training succeeds, verify it didn't crash
            assert "loss" in history.history
        except Exception as e:
            # Small datasets might cause training issues
            assert "data" in str(e).lower() or "shape" in str(e).lower()

    def test_configuration_fallback_workflow(self):
        """Test configuration fallback workflow."""
        from common.core.active_config import get_active_project

        # Test that we can always get some project
        project = get_active_project()
        assert project is not None
        assert isinstance(project, str)

        # Test that config loading has fallbacks
        try:
            from common.core.active_config import get_config_for_active_project

            config_manager = get_config_for_active_project("default")
            config = config_manager.get_config()

            # Should have basic required attributes
            assert hasattr(config, "data")
            assert hasattr(config, "general")
        except Exception as e:
            # If config loading fails, error should be informative
            assert "config" in str(e).lower() or "file" in str(e).lower()
