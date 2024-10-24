# interpreter.py

import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier
import shap
import pickle
import logging
import hashlib


class Prediction:
    """
    This class provides methods for evaluating and making predictions using a trained machine learning model.

    Methods:
        calculate_baseline_scores(y_true):
            Calculates the baseline accuracy given a set of true labels.

        evaluate_model(model, test_inputs, test_labels):
            Evaluates the model on test data.

        make_predictions(model, test_inputs):
            Makes predictions on test data using the trained model.
    """

    @staticmethod
    def calculate_baseline_scores(y_true):
        """
        Calculate the baseline accuracy given a set of true labels.

        Args:
            y_true (numpy.ndarray): An array of true labels.

        Returns:
            baseline_accuracy (float): The baseline accuracy.
            baseline_precision (float): The baseline precision.
            baseline_recall (float): The baseline recall.
            baseline_f1 (float): The baseline f1 score.
        """

        # Create a dummy classifier that will predict the most frequent class
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(y_true, y_true)
        baseline_preds = dummy_clf.predict(y_true)

        # Calculate accuracy, precision, recall, and f1 score
        baseline_accuracy = accuracy_score(y_true, baseline_preds)
        baseline_precision = precision_score(y_true, baseline_preds, average="macro")
        baseline_recall = recall_score(y_true, baseline_preds, average="macro")
        baseline_f1 = f1_score(y_true, baseline_preds, average="macro")

        return baseline_accuracy, baseline_precision, baseline_recall, baseline_f1

    @staticmethod
    def evaluate_model(model, test_inputs, test_labels):
        """
        Evaluate the model on test data.

        Args:
            model (Model): A trained TensorFlow model.
            test_inputs (numpy.ndarray): Test input data.
            test_labels (numpy.ndarray): Test labels.

        Returns:
            test_loss (float): The value of the test loss for the input data.
            test_acc (float): The value of the test accuracy for the input data.
            test_auc (float): The value of the test AUC for the input data.
        """

        # Evaluate the model on test data
        test_loss, test_acc, test_auc = model.evaluate(test_inputs, test_labels)
        return test_loss, test_acc, test_auc

    @staticmethod
    def make_predictions(model, test_inputs):
        """
        Make predictions on test data using the trained model.

        Args:
            model (Model): A trained TensorFlow model.
            test_inputs (numpy.ndarray): Test input data.

        Returns:
            y_pred (numpy.ndarray): An array of predicted probabilities for the input data.
            y_pred_binary (numpy.ndarray): An array of binary predictions for the input data.
        """

        # Make predictions using the trained model
        y_pred = model.predict(test_inputs)

        # Convert the predicted probabilities into binary predictions
        y_pred_binary = np.argmax(y_pred, axis=1)

        return y_pred, y_pred_binary


class Interpreter:
    """
    A class to handle model interpretation using SHAP.
    """

    def __init__(
        self,
        config,
        model,
        test_data,
        test_labels,
        label_encoder,
        reference_data,
        path_manager,
    ):
        """
        Initializes the Interpreter with the given configuration and model.

        Parameters:
        - config (Config): The configuration object.
        - model (object): The best model to be interpreted.
        - test_data (numpy.ndarray): The test data used during model training.
        - test_labels (numpy.ndarray): The labels for the test data.
        - label_encoder (LabelEncoder): The label encoder used during training.
        - reference_data (numpy.ndarray): The reference data used during model training.
        """
        self.config = config
        self.model = model  # Best model for SHAP analysis
        self.test_data = test_data  # Best model's test data
        self.test_labels = test_labels  # Best model's test labels
        self.label_encoder = label_encoder
        self.reference_data = reference_data  # Best model's reference data
        self.path_manager = path_manager

        self.shap_dir = self.path_manager.get_visualization_directory(
            subfolder="SHAP",
        )

        # Set the filepath for SHAP values
        self.shap_values_filepath = os.path.join(self.shap_dir, "shap_values.pkl")

    def compute_or_load_shap_values(self):
        """
        Compute or load SHAP values for model interpretation based on the configuration.

        Returns:
        - tuple: A tuple containing the SHAP values and the corresponding SHAP test data.
        """
        shap_values = None

        if self.config.FeatureImportanceAndVisualizations.load_SHAP:
            # Attempt to load SHAP values from disk
            try:
                shap_values = self.load_shap_values()
                squeezed_test_data = (
                    self.test_data
                    if self.test_data.ndim <= 2
                    else np.squeeze(self.test_data, axis=1)
                )
                logging.info("Loaded SHAP values and data from disk.")
            except Exception as e:
                logging.error(f"Error loading SHAP values: {e}")
                logging.info("Computing SHAP values instead.")
                shap_values, squeezed_test_data = self.compute_shap_values()
                self.save_shap_values(shap_values)
        else:
            # Compute SHAP values
            shap_values, squeezed_test_data = self.compute_shap_values()
            self.save_shap_values(shap_values)
            logging.info("Computed and saved SHAP values.")

        return shap_values, squeezed_test_data

    def compute_shap_values(self):
        """
        Compute SHAP values for model interpretation.

        Returns:
        - tuple: A tuple containing the SHAP values and the corresponding SHAP test data.
        """

        # Access the model type from the configuration
        model_type = self.config.DataParameters.GeneralSettings.model_type.lower()

        # Determine the explainer to use based on the model type
        if model_type in ["mlp", "cnn"]:
            # For neural network models, use GradientExplainer
            explainer = shap.GradientExplainer(self.model, self.reference_data)
        elif model_type in ["xgboost", "randomforest"]:
            # For tree-based models, use TreeExplainer
            explainer = shap.TreeExplainer(self.model)
        else:
            # For linear models, use LinearExplainer
            explainer = shap.LinearExplainer(self.model, self.reference_data)

        # Compute SHAP values
        shap_values = explainer.shap_values(self.test_data)

        # Adjust SHAP values and test data shapes if necessary
        if isinstance(shap_values, list):
            squeezed_shap_values = [
                np.squeeze(val, axis=1) if val.ndim > 3 else val for val in shap_values
            ]
        else:
            squeezed_shap_values = (
                np.squeeze(shap_values, axis=1) if shap_values.ndim > 3 else shap_values
            )

        # Convert the SHAP values to a list of arrays for compatibility with the rest of the code
        squeezed_shap_values = [
            squeezed_shap_values[:, :, i] for i in range(squeezed_shap_values.shape[2])
        ]

        squeezed_test_data = (
            self.test_data
            if self.test_data.ndim <= 2
            else np.squeeze(self.test_data, axis=1)
        )

        return squeezed_shap_values, squeezed_test_data

    def save_shap_values(self, shap_values):
        """
        Save SHAP values, model metadata, and the data used during SHAP computation.

        Args:
            shap_values (list or numpy.ndarray): The SHAP values to be saved.
        """
        # Collect model and data metadata
        model_weights_hash = self._get_model_weights_hash()
        metadata = {
            "model_type": self.config.DataParameters.GeneralSettings.model_type,
            "model_config": (
                self.model.get_config() if hasattr(self.model, "get_config") else None
            ),
            "model_weights_hash": model_weights_hash,
            "test_data_hash": compute_sha256_hash(self.test_data.tobytes()),
            "reference_data_hash": compute_sha256_hash(self.reference_data.tobytes()),
        }

        # Save SHAP values, metadata, and the data
        with open(self.shap_values_filepath, "wb") as f:
            pickle.dump(
                {
                    "shap_values": shap_values,
                    "metadata": metadata,
                    "reference_data": self.reference_data,
                    "test_data": self.test_data,
                },
                f,
            )
        logging.info(
            f"SHAP values, metadata, and data saved to {self.shap_values_filepath}"
        )

    def load_shap_values(self):
        """
        Load previously saved SHAP values along with the reference data and test data used during SHAP computation.

        Returns:
            The loaded SHAP values.
        """
        with open(self.shap_values_filepath, "rb") as f:
            data = pickle.load(f)

        loaded_shap_values = data["shap_values"]
        saved_metadata = data["metadata"]
        loaded_reference_data = data["reference_data"]
        loaded_test_data = data["test_data"]

        # Collect current model metadata and data hashes before updating data
        model_weights_hash = self._get_model_weights_hash()
        current_metadata = {
            "model_type": self.config.DataParameters.GeneralSettings.model_type,
            "model_config": (
                self.model.get_config() if hasattr(self.model, "get_config") else None
            ),
            "model_weights_hash": model_weights_hash,
            "test_data_hash": (
                compute_sha256_hash(self.test_data.tobytes())
                if self.test_data is not None
                else None
            ),
            "reference_data_hash": (
                compute_sha256_hash(self.reference_data.tobytes())
                if self.reference_data is not None
                else None
            ),
        }

        # Verify model metadata
        model_match = (
            saved_metadata["model_weights_hash"]
            == current_metadata["model_weights_hash"]
        )

        # Verify data metadata
        test_data_match = (
            saved_metadata["test_data_hash"] == current_metadata["test_data_hash"]
        )
        reference_data_match = (
            saved_metadata["reference_data_hash"]
            == current_metadata["reference_data_hash"]
        )

        # Perform consistency checks
        if not model_match:
            logging.warning(
                "The model used for SHAP computation does not match the current model."
            )
        if not test_data_match:
            logging.warning(
                "The test data used for SHAP computation does not match the current test data."
            )
        if not reference_data_match:
            logging.warning(
                "The reference data used for SHAP computation does not match the current reference data."
            )

        if model_match and test_data_match and reference_data_match:
            logging.info(
                "The model and data used for SHAP computation match the current best model and data."
            )
        else:
            logging.warning("Consider recomputing SHAP values for accurate results.")

        # Now update self.test_data and self.reference_data
        self.test_data = loaded_test_data
        self.reference_data = loaded_reference_data

        logging.info(f"SHAP values and data loaded from {self.shap_values_filepath}")
        return loaded_shap_values

    def _get_model_weights_hash(self):
        """
        Get a SHA-256 hash of the model's weights for consistency checks.

        Returns:
            str: The hexadecimal SHA-256 hash of the model's weights.
        """
        if hasattr(self.model, "get_weights"):
            # For Keras models
            weights = self.model.get_weights()
            weights_bytes = b"".join([w.tobytes() for w in weights])
            return compute_sha256_hash(weights_bytes)
        else:
            # For other models like scikit-learn models
            model_bytes = pickle.dumps(self.model)
            return compute_sha256_hash(model_bytes)


def compute_sha256_hash(data_bytes):
    """
    Compute the SHA-256 hash of the given bytes.

    Args:
        data_bytes (bytes): The data to hash.

    Returns:
        str: The hexadecimal SHA-256 hash of the data.
    """
    return hashlib.sha256(data_bytes).hexdigest()
