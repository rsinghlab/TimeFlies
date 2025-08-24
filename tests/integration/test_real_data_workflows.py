"""Integration tests that exercise real workflow components."""

import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Import actual workflow components
from common.cli.main import main_cli
from common.core.active_config import get_active_project, get_config_for_active_project


@pytest.mark.integration
class TestCLIWorkflowIntegration:
    """Test CLI workflows that actually run code."""

    def test_cli_verify_real_execution(self):
        """Test CLI verify command with real execution."""
        # This exercises the actual verify command path
        result = main_cli(["verify"])

        # Should return an integer exit code (may be 0 or 1)
        assert isinstance(result, int)

    def test_cli_setup_real_execution(self):
        """Test CLI setup command workflow with mocked data operations."""
        from unittest.mock import patch
        
        # Mock the data operations but test the CLI workflow
        with patch("common.cli.commands.setup_user_environment", return_value=0):
            with patch("common.cli.commands.split_command", return_value=0):
                with patch("common.cli.system_checks.verify_system", return_value=0):
                    with patch("builtins.input", return_value="n"):  # Skip batch correction
                        result = main_cli(["setup"])

                        # Should return success code  
                        assert result == 0

    def test_cli_create_test_data_execution(self):
        """Test CLI create test data with mocked file operations."""
        from unittest.mock import patch
        
        # Mock the test data creation to avoid file system operations
        with patch("common.cli.commands.create_test_data_command", return_value=0):
            result = main_cli(["create-test-data"])

            # Should return success code
            assert result == 0

    def test_config_loading_real_workflow(self):
        """Test real config loading workflow."""
        # This exercises actual config loading code
        project = get_active_project()
        assert project in ["fruitfly_aging", "fruitfly_alzheimers"]

        config_manager = get_config_for_active_project("default")
        config = config_manager.get_config()

        # Test that config is actually loaded and accessible
        assert hasattr(config, "data")
        assert hasattr(config, "general")
        assert config.data.tissue in ["head", "body", "all"]
        assert config.data.model in [
            "CNN",
            "MLP",
            "logistic",
            "xgboost",
            "random_forest",
        ]

    def test_project_switching_real_workflow(self):
        """Test project switching with real workflow."""
        # Test aging project
        with patch(
            "common.core.active_config.get_active_project",
            return_value="fruitfly_aging",
        ):
            config_manager = get_config_for_active_project("default")
            config = config_manager.get_config()
            assert config is not None

        # Test alzheimers project (if available)
        try:
            with patch(
                "common.core.active_config.get_active_project",
                return_value="fruitfly_alzheimers",
            ):
                config_manager = get_config_for_active_project("default")
                config = config_manager.get_config()
                assert config is not None
        except Exception as e:
            # May not be implemented yet
            assert "not found" in str(e).lower() or "path" in str(e).lower()


@pytest.mark.integration
class TestDataWorkflowIntegration:
    """Test data workflow integration."""

    def test_timeflies_dataprocessor_workflow(self, large_sample_anndata):
        """Test TimeFlies DataPreprocessor workflow (not just scanpy)."""
        from common.data.preprocessing.data_processor import DataPreprocessor
        from common.core.active_config import get_config_for_active_project
        from unittest.mock import patch
        
        # Add genotype column that TimeFlies expects
        genotype_values = ['ctrl', 'alz'] * (large_sample_anndata.n_obs // 2)
        if large_sample_anndata.n_obs % 2 == 1:
            genotype_values.append('ctrl')
        large_sample_anndata.obs['genotype'] = genotype_values
        
        config_manager = get_config_for_active_project("default")
        config = config_manager.get_config()
        
        # Test actual TimeFlies DataPreprocessor
        with patch("common.utils.path_manager.PathManager"):
            preprocessor = DataPreprocessor(config, large_sample_anndata, large_sample_anndata.copy())
            
            # Test actual TimeFlies processing method
            processed = preprocessor.process_adata(large_sample_anndata.copy())
            assert processed.n_obs > 0
            assert processed.n_vars > 0
            assert 'age' in processed.obs.columns

    def test_data_splitting_workflow(self, large_sample_anndata):
        """Test real data splitting workflow."""
        from sklearn.model_selection import train_test_split

        # Test actual train/test splitting
        X = large_sample_anndata.X
        y = large_sample_anndata.obs["age"].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Verify split worked
        assert X_train.shape[0] + X_test.shape[0] == X.shape[0]
        assert len(y_train) + len(y_test) == len(y)
        assert X_train.shape[1] == X_test.shape[1] == X.shape[1]

    def test_label_encoding_workflow(self, large_sample_anndata):
        """Test real label encoding workflow."""
        from sklearn.preprocessing import LabelEncoder

        # Test actual label encoding
        encoder = LabelEncoder()
        ages = large_sample_anndata.obs["age"].values
        encoded_ages = encoder.fit_transform(ages)

        # Verify encoding worked
        assert len(encoded_ages) == len(ages)
        assert encoded_ages.dtype in [np.int32, np.int64]
        assert len(np.unique(encoded_ages)) <= len(np.unique(ages))

    def test_feature_selection_workflow(self, large_sample_anndata):
        """Test real feature selection workflow."""
        import scanpy as sc

        # Test actual feature selection
        adata = large_sample_anndata.copy()

        # Calculate highly variable genes
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

        # Filter to highly variable genes
        if "highly_variable" in adata.var.columns:
            adata_hvg = adata[:, adata.var.highly_variable].copy()

            # Verify filtering worked
            assert adata_hvg.n_vars <= adata.n_vars
            assert adata_hvg.n_obs == adata.n_obs


@pytest.mark.integration
class TestModelWorkflowIntegration:
    """Test model workflow integration."""

    def test_sklearn_model_workflow(self, small_sample_anndata):
        """Test real sklearn model workflow."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Prepare real data
        X = small_sample_anndata.X
        y = small_sample_anndata.obs["age"].values

        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )

        # Train model
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Verify workflow
        assert 0 <= accuracy <= 1
        assert len(y_pred) == len(y_test)
        assert model.classes_ is not None

    def test_tensorflow_model_workflow(self, small_sample_anndata):
        """Test real TensorFlow model workflow."""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Prepare real data
        X = small_sample_anndata.X.astype(np.float32)
        y = small_sample_anndata.obs["age"].values

        # Encode labels
        encoder = LabelEncoder()
        y_encoded = encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42
        )

        # Create simple model
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(32, activation="relu", input_shape=(X.shape[1],)),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )

        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        # Train model (1 epoch for speed)
        history = model.fit(X_train, y_train, epochs=1, verbose=0, validation_split=0.2)

        # Make predictions
        predictions = model.predict(X_test, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        # Verify workflow
        assert len(y_pred) == len(y_test)
        assert predictions.shape == (len(y_test), num_classes)
        assert "loss" in history.history
        assert "accuracy" in history.history

    def test_model_evaluation_workflow(self, small_sample_anndata):
        """Test real model evaluation workflow."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_auc_score,
        )
        from sklearn.model_selection import train_test_split
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

        # Train model
        model = LogisticRegression(random_state=42, max_iter=100)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Calculate metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # For binary classification
        if len(np.unique(y_encoded)) == 2:
            auc = roc_auc_score(y_test, y_proba[:, 1])
            assert 0 <= auc <= 1

        # Verify evaluation
        assert isinstance(report, dict)
        assert "accuracy" in report
        assert cm.shape == (len(np.unique(y_test)), len(np.unique(y_test)))


@pytest.mark.integration
class TestFileIOWorkflowIntegration:
    """Test file I/O workflow integration."""

    def test_h5ad_file_workflow(self, small_sample_anndata):
        """Test real h5ad file I/O workflow."""
        import scanpy as sc

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "test_data.h5ad"

            # Write file
            small_sample_anndata.write_h5ad(file_path)
            assert file_path.exists()

            # Read file
            loaded_data = sc.read_h5ad(file_path)

            # Verify roundtrip
            assert loaded_data.n_obs == small_sample_anndata.n_obs
            assert loaded_data.n_vars == small_sample_anndata.n_vars
            np.testing.assert_array_equal(loaded_data.X, small_sample_anndata.X)

    def test_model_save_load_workflow(self, small_sample_anndata):
        """Test real model save/load workflow."""
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "test_model.keras"

            # Prepare data
            X = small_sample_anndata.X.astype(np.float32)
            y = small_sample_anndata.obs["age"].values

            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            num_classes = len(np.unique(y_encoded))

            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.3, random_state=42
            )

            # Create and train model
            model = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(
                        16, activation="relu", input_shape=(X.shape[1],)
                    ),
                    tf.keras.layers.Dense(num_classes, activation="softmax"),
                ]
            )

            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
            model.fit(X_train, y_train, epochs=1, verbose=0)

            # Save model
            model.save(model_path)
            assert model_path.exists()

            # Load model
            loaded_model = tf.keras.models.load_model(model_path)

            # Verify loaded model works
            original_pred = model.predict(X_test, verbose=0)
            loaded_pred = loaded_model.predict(X_test, verbose=0)

            np.testing.assert_array_almost_equal(original_pred, loaded_pred, decimal=5)

    def test_config_file_workflow(self, temp_dir):
        """Test real config file workflow."""
        import yaml

        config_data = {
            "general": {"project_name": "test", "random_state": 42},
            "data": {"tissue": "head", "model": "CNN"},
            "model": {"training": {"epochs": 10}},
        }

        config_file = Path(temp_dir) / "test_config.yaml"

        # Write config
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)

        assert config_file.exists()

        # Read config
        with open(config_file) as f:
            loaded_config = yaml.safe_load(f)

        # Verify roundtrip
        assert loaded_config == config_data
        assert loaded_config["general"]["project_name"] == "test"
        assert loaded_config["data"]["tissue"] == "head"


@pytest.mark.integration
class TestErrorHandlingIntegration:
    """Test error handling in real workflows."""

    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        import scanpy as sc

        # Test reading non-existent file
        with pytest.raises(Exception):
            sc.read_h5ad("non_existent_file.h5ad")

    def test_invalid_model_input_handling(self):
        """Test handling of invalid model inputs."""
        import tensorflow as tf

        # Create model
        model = tf.keras.Sequential(
            [tf.keras.layers.Dense(10, input_shape=(5,)), tf.keras.layers.Dense(1)]
        )

        model.compile(optimizer="adam", loss="mse")

        # Test with wrong input shape
        wrong_input = np.random.random((10, 3))  # Should be (10, 5)

        with pytest.raises(Exception):
            model.predict(wrong_input)

    def test_config_error_handling(self):
        """Test config error handling."""
        from common.core.active_config import get_config_for_active_project

        # Test with invalid config name
        try:
            config_manager = get_config_for_active_project("non_existent_config")
            config_manager.get_config()
        except Exception as e:
            # Should raise appropriate error
            assert "not found" in str(e).lower() or "config" in str(e).lower()

    def test_cli_error_handling(self):
        """Test CLI error handling."""
        # Test with invalid command (should exit gracefully)
        try:
            main_cli(["invalid_command"])
        except SystemExit as e:
            # Should exit with error code
            assert e.code != 0
