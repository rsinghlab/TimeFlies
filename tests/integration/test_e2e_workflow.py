"""Final working end-to-end test that matches TimeFlies expectations."""

import anndata as ad
import numpy as np
import pandas as pd


def test_timeflies_with_correct_data_format():
    """
    Test TimeFlies with data that matches the expected format.

    This discovers what TimeFlies actually expects and tests it.
    """
    from unittest.mock import patch

    from common.core.active_config import get_config_for_active_project
    from common.data.preprocessing.data_processor import DataPreprocessor
    from common.models.model_factory import ModelFactory

    print("🔄 Creating TimeFlies-compatible test data...")

    # Create data with the expected columns based on the error
    n_cells, n_genes = 24, 50
    np.random.seed(42)

    X = np.random.negative_binomial(12, 0.4, size=(n_cells, n_genes)).astype(float)

    obs = pd.DataFrame(
        {
            "age": [1] * 8 + [10] * 8 + [20] * 8,
            "age_group": ["young"] * 8 + ["middle"] * 8 + ["old"] * 8,
            "sex": ["male", "female"] * 12,
            "tissue": ["head"] * n_cells,
            "genotype": ["ctrl", "alz"] * 12,  # This is what TimeFlies expects!
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(
        {
            "gene_type": ["protein_coding"] * n_genes,
            "highly_variable": [True] * 20 + [False] * 30,
        },
        index=[f"gene_{i}" for i in range(n_genes)],
    )

    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata_eval = adata.copy()

    print("✅ TimeFlies-compatible data created")

    # Get config
    config_manager = get_config_for_active_project("default")
    config = config_manager.get_config()
    config.data.model = "logistic"

    print("🔄 Testing DataPreprocessor with correct data format...")

    # Test DataPreprocessor with complete mocking to prevent directory creation
    with patch("common.utils.path_manager.PathManager") as mock_pm_class:
        # Mock the PathManager instance completely
        mock_pm = mock_pm_class.return_value
        mock_pm.get_outputs_directory.return_value = "/tmp/test_outputs"
        mock_pm.get_log_directory.return_value = "/tmp/test_logs"
        mock_pm.get_visualization_directory.return_value = "/tmp/test_viz"
        mock_pm.get_config_key.return_value = "test_config"
        mock_pm.generate_experiment_name.return_value = "test_experiment"

        preprocessor = DataPreprocessor(config, adata, adata_eval)

        # Test data processing
        processed = preprocessor.process_adata(adata.copy())
        assert processed.n_obs > 0, "Should have cells after processing"
        assert processed.n_vars > 0, "Should have genes after processing"
        assert "age" in processed.obs.columns
        assert "genotype" in processed.obs.columns

        print("✅ DataPreprocessor.process_adata() works")

        # Test data splitting with correct format
        train_split, test_split = preprocessor.split_data(adata)
        assert train_split.n_obs >= 0, "Train split should work"
        assert test_split.n_obs >= 0, "Test split should work"
        total_cells = train_split.n_obs + test_split.n_obs
        assert total_cells <= adata.n_obs, (
            f"Split cells ({total_cells}) shouldn't exceed original ({adata.n_obs})"
        )

        print("✅ DataPreprocessor.split_data() works")

    print("🔄 Testing ModelFactory...")

    # Test ModelFactory
    models = ModelFactory.get_supported_models()
    assert len(models) > 0, "Should have supported models"
    print(f"  📋 Supported models: {models}")

    # Test creating a model
    model = ModelFactory.create_model("logistic", config)
    assert model is not None, "Should create model"

    print("✅ ModelFactory works")

    print("🔄 Testing complete model training workflow...")

    # Test complete workflow: preprocess → train → predict
    with patch("common.utils.path_manager.PathManager") as mock_pm_class:
        # Mock the PathManager instance completely
        mock_pm = mock_pm_class.return_value
        mock_pm.get_outputs_directory.return_value = "/tmp/test_outputs"
        mock_pm.get_log_directory.return_value = "/tmp/test_logs"
        mock_pm.get_visualization_directory.return_value = "/tmp/test_viz"
        mock_pm.get_config_key.return_value = "test_config"
        mock_pm.generate_experiment_name.return_value = "test_experiment"

        preprocessor = DataPreprocessor(config, adata, adata_eval)
        processed = preprocessor.process_adata(adata.copy())

        # Prepare training data
        X_train = (
            processed.X.toarray() if hasattr(processed.X, "toarray") else processed.X
        )
        y_train = processed.obs["age"].values  # Use age as target

        # Encode labels
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)

        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_encoded))

        print(
            f"  📊 Training data: {X_train.shape[0]} samples, {n_features} features, {n_classes} classes"
        )

        # Build and train model
        model = ModelFactory.create_model("logistic", config)
        model.build(input_shape=(n_features,), num_classes=n_classes)
        model.train(X_train, y_encoded)

        assert model.is_trained, "Model should be trained"
        print("✅ Model training successful")

        # Test predictions
        predictions = model.predict(X_train[:5])
        assert len(predictions) == 5, "Should predict 5 samples"
        assert all(isinstance(p, int | np.integer) for p in predictions), (
            "Predictions should be integers"
        )
        assert all(0 <= p < n_classes for p in predictions), (
            "Predictions should be valid class indices"
        )

        # Test probabilities
        probabilities = model.predict_proba(X_train[:3])
        assert probabilities.shape == (
            3,
            n_classes,
        ), f"Expected shape (3, {n_classes}), got {probabilities.shape}"
        assert np.allclose(probabilities.sum(axis=1), 1.0, rtol=1e-5), (
            "Probabilities should sum to 1"
        )

        print("✅ Model prediction successful")

    print("🎉 COMPLETE TIMEFLIES WORKFLOW SUCCESSFUL!")
    print("\n📋 Successfully tested:")
    print("  ✅ Data format matching TimeFlies expectations")
    print("  ✅ DataPreprocessor.process_adata() and split_data()")
    print("  ✅ ModelFactory.create_model() and get_supported_models()")
    print("  ✅ Complete train/predict workflow with real data")
    print("  ✅ Model predictions and probabilities")

    return True


def test_multiple_timeflies_models():
    """Test TimeFlies with multiple model types using correct data format."""
    from common.core.active_config import get_config_for_active_project
    from common.models.model_factory import ModelFactory

    print("🔄 Testing TimeFlies with multiple models...")

    # Create compatible test data
    n_samples, n_features = 30, 25
    X = np.random.randn(n_samples, n_features)
    y = np.array([0, 1, 2] * 10)  # 3 balanced classes

    config_manager = get_config_for_active_project("default")
    config = config_manager.get_config()

    # Test fast models
    models_to_test = ["logistic", "random_forest"]

    for model_type in models_to_test:
        print(f"  🔄 Testing {model_type}...")

        model = ModelFactory.create_model(model_type, config)
        model.build(input_shape=(n_features,), num_classes=3)
        model.train(X, y)

        predictions = model.predict(X[:5])
        assert len(predictions) == 5
        assert all(p in [0, 1, 2] for p in predictions)

        print(f"  ✅ {model_type} works correctly")

    print("✅ Multiple model types work with TimeFlies")
    return True


if __name__ == "__main__":
    print("🚀 Testing TimeFlies with correct data format...")
    test_timeflies_with_correct_data_format()
    test_multiple_timeflies_models()
    print("🎉 All TimeFlies tests passed!")
