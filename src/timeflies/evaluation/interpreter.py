# interpreter.py

import os
import numpy as np
from sklearn.dummy import DummyClassifier
import shap
import pickle
import logging
import hashlib
from sklearn.preprocessing import label_binarize
import pandas as pd
import json
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


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
        - path_manager (PathManager): The path manager object for directory paths.
        """
        self.config = config
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.reference_data = reference_data
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

        if getattr(self.config.feature_importance, 'load_SHAP', False):
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

        # Squeeze the test data if necessary
        squeezed_test_data = (
            self.test_data
            if self.test_data.ndim <= 2
            else np.squeeze(self.test_data, axis=1)
        )

        # Access the model type from the configuration
        model_type = getattr(self.config.data, 'model_type', 'CNN').lower()

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


        # Adjust SHAP values and test data shapes based on the system type
        device = getattr(self.config.device, 'processor', 'GPU').lower()
        if device == "m":
            # Adjust SHAP values for macOS
            if isinstance(shap_values, list):
                squeezed_shap_values = [
                    np.squeeze(val, axis=1) if val.ndim >= 3 else val
                    for val in shap_values
                ]
            else:
                squeezed_shap_values = (
                    np.squeeze(shap_values, axis=1)
                    if shap_values.ndim >= 3
                    else shap_values
                )

        else:
            # Adjust SHAP values for Windows
            if isinstance(shap_values, list):
                squeezed_shap_values = [
                    np.squeeze(val, axis=1) if val.ndim > 3 else val
                    for val in shap_values
                ]
            else:
                squeezed_shap_values = (
                    np.squeeze(shap_values, axis=1)
                    if shap_values.ndim > 3
                    else shap_values
                )

            # Convert the SHAP values to a list of arrays for compatibility with the rest of the code
            squeezed_shap_values = [
                squeezed_shap_values[:, :, i]
                for i in range(squeezed_shap_values.shape[2])
            ]

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
            "model_type": getattr(self.config.data, 'model_type', 'CNN'),
            "model_config": (
                self.model.get_config() if hasattr(self.model, "get_config") else None
            ),
            "model_weights_hash": model_weights_hash,
            "test_data_hash": self.compute_sha256_hash(self.test_data.tobytes()),
            "reference_data_hash": self.compute_sha256_hash(
                self.reference_data.tobytes()
            ),
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
            "model_type": getattr(self.config.data, 'model_type', 'CNN'),
            "model_config": (
                self.model.get_config() if hasattr(self.model, "get_config") else None
            ),
            "model_weights_hash": model_weights_hash,
            "test_data_hash": (
                self.compute_sha256_hash(self.test_data.tobytes())
                if self.test_data is not None
                else None
            ),
            "reference_data_hash": (
                self.compute_sha256_hash(self.reference_data.tobytes())
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
            return self.compute_sha256_hash(weights_bytes)
        else:
            # For other models like scikit-learn models
            model_bytes = pickle.dumps(self.model)
            return self.compute_sha256_hash(model_bytes)

    def compute_sha256_hash(self, data_bytes):
        """
        Compute the SHA-256 hash of the given bytes.

        Args:
            data_bytes (bytes): The data to hash.

        Returns:
            str: The hexadecimal SHA-256 hash of the data.
        """
        return hashlib.sha256(data_bytes).hexdigest()


class Metrics:
    """
    A class to handle evaluation and saving of model performance metrics.

    This class evaluates the model's performance, calculates metrics, and saves the results.
    """

    def __init__(
        self,
        config,
        model,
        test_inputs,
        test_labels,
        label_encoder,
        path_manager,
    ):
        """
        Initializes the Metrics class with the given configuration and results.

        Parameters:
        - config (ConfigHandler): A configuration handler object with nested configuration attributes.
        - model (object): The trained model.
        - test_inputs (numpy.ndarray): The test data.
        - test_labels (numpy.ndarray): The labels for the test data.
        - label_encoder (LabelEncoder): The label encoder used during training.
        - path_manager (PathManager): The path manager object for directory paths.
        """
        self.config = config
        self.model = model
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.path_manager = path_manager

        # Determine the main output directory using PathManager
        self.output_dir = self.path_manager.get_visualization_directory()

    def _evaluate_model_performance(self):
        """
        Evaluate the model on the test data and store performance metrics.
        """
        model_type = getattr(self.config.data, 'model_type', 'CNN').lower()
        if model_type in ["mlp", "cnn"]:
            test_loss, test_acc, test_auc = Prediction.evaluate_model(
                self.model, self.test_inputs, self.test_labels
            )
            logger.info(f"Eval accuracy: {test_acc}")
            logger.info(f"Eval loss: {test_loss}")
            logger.info(f"Eval AUC: {test_auc}")

        if model_type in ["mlp", "cnn"]:
            self.y_pred = self.model.predict(self.test_inputs)
        else:
            self.y_pred = self.model.predict_proba(self.test_inputs)

        # Convert predictions and true labels to class indices
        self.y_pred_class = np.argmax(self.y_pred, axis=1)
        self.y_true_class = np.argmax(self.test_labels, axis=1)

    def _calculate_and_save_metrics(self):
        """
        Calculate, log, and save various performance metrics.
        """
        # Calculate metrics
        accuracy = accuracy_score(self.y_true_class, self.y_pred_class)
        precision = precision_score(
            self.y_true_class, self.y_pred_class, average="macro"
        )
        recall = recall_score(self.y_true_class, self.y_pred_class, average="macro")
        f1 = f1_score(self.y_true_class, self.y_pred_class, average="macro")

        # Compute ROC-AUC score
        y_true_binary = label_binarize(
            self.y_true_class, classes=np.unique(self.y_true_class)
        )
        y_pred_prob = self.y_pred

        n_classes = len(np.unique(self.y_true_class))
        if n_classes == 2:  # Binary classification
            y_pred_prob_positive = y_pred_prob[:, 1]
            auc_score = roc_auc_score(
                y_true_binary, y_pred_prob_positive, average="macro", multi_class="ovo"
            )
        else:  # Multi-class classification
            auc_score = roc_auc_score(
                y_true_binary, y_pred_prob, average="macro", multi_class="ovo"
            )

        # Log the classification report
        class_labels = self.label_encoder.classes_
        logger.info("Classification Report:")
        logger.info(
            classification_report(
                self.y_true_class, self.y_pred_class, target_names=class_labels
            )
        )

        # Log performance metrics
        logger.info(
            f"Test Accuracy: {accuracy:.4%}, Test Precision: {precision:.4%}, Test Recall: {recall:.4%}, Test F1: {f1:.4%}, Test AUC: {auc_score:.4%}"
        )

        # Calculate baseline metrics
        (
            baseline_accuracy,
            baseline_precision,
            baseline_recall,
            baseline_f1,
        ) = Prediction.calculate_baseline_scores(self.y_true_class)

        # Log baseline metrics
        logger.info(
            f"Baseline Accuracy: {baseline_accuracy:.4%}, Baseline Precision: {baseline_precision:.4%}, "
            f"Baseline Recall: {baseline_recall:.4%}, Baseline F1: {baseline_f1:.4%}"
        )

        # Save metrics as JSON
        self.save_metrics_as_json(
            test_accuracy=accuracy,
            test_precision=precision,
            test_recall=recall,
            test_f1=f1,
            test_auc=auc_score,
            baseline_accuracy=baseline_accuracy,
            baseline_precision=baseline_precision,
            baseline_recall=baseline_recall,
            baseline_f1=baseline_f1,
            file_name="Stats.JSON",
        )

    def save_metrics_as_json(
        self,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
        test_auc,
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1,
        file_name,
    ):
        """
        Saves the provided metrics to a JSON file.

        Parameters:
            test_accuracy (float): Test accuracy.
            test_precision (float): Test precision.
            test_recall (float): Test recall.
            test_f1 (float): Test F1 score.
            test_auc (float): Test AUC.
            baseline_accuracy (float): Baseline accuracy.
            baseline_precision (float): Baseline precision.
            baseline_recall (float): Baseline recall.
            baseline_f1 (float): Baseline F1 score.
            file_name (str): The name of the file to save the metrics in.
        """
        # Function to format the metrics as percentages
        format_percent = lambda x: f"{x * 100:.2f}%"

        # Construct the metrics dictionary with formatted values
        metrics = {
            "Test": {
                "Accuracy": format_percent(test_accuracy),
                "Precision": format_percent(test_precision),
                "Recall": format_percent(test_recall),
                "F1": format_percent(test_f1),
                "AUC": format_percent(test_auc),
            },
            "Baseline": {
                "Accuracy": format_percent(baseline_accuracy),
                "Precision": format_percent(baseline_precision),
                "Recall": format_percent(baseline_recall),
                "F1": format_percent(baseline_f1),
            },
        }

        # Save metrics to main analysis folder
        os.makedirs(self.output_dir, exist_ok=True)
        output_file_path = os.path.join(self.output_dir, file_name)
        with open(output_file_path, "w") as file:
            json.dump(metrics, file, indent=4)

        # If interpretable is True, save an additional copy to the SHAP directory
        if getattr(self.config.feature_importance, 'run_interpreter', True):
            shap_dir = self.path_manager.get_visualization_directory(subfolder="SHAP")
            os.makedirs(shap_dir, exist_ok=True)
            shap_output_file_path = os.path.join(shap_dir, "Stats.JSON")
            with open(shap_output_file_path, "w") as file:
                json.dump(metrics, file, indent=4)

    def save_predictions_to_csv(self, file_name_template="{}_{}_predictions.csv"):
        """
        Save the predicted and actual labels to a CSV file, naming it based on the training and test data configuration.
        Only the method specified in the config (e.g., 'sex' or 'tissue') will be used to name the file.

        Args:
            file_name_template (str): A template for naming the file with placeholders for train/test attributes.

        Returns:
            None
        """
        # Convert predictions and true labels to class indices if not already done
        if getattr(self.config.feature_importance, 'save_predictions', False):

            if not hasattr(self, "y_pred_class"):
                self.y_pred_class = np.argmax(self.y_pred, axis=1)
                self.y_true_class = np.argmax(self.test_labels, axis=1)

            class_names = self.label_encoder.classes_

            # Map the predicted and actual class indices back to the class names
            y_pred_names = [class_names[i] for i in self.y_pred_class]
            y_true_names = [class_names[i] for i in self.y_true_class]

            # Create a DataFrame with predicted and actual labels (class names)
            df_predictions = pd.DataFrame(
                {"Predicted": y_pred_names, "Actual": y_true_names}
            )

            # Determine the relevant train and test attributes based on the method
            method = getattr(self.config.data.train_test_split, 'method', 'random')  # This could be 'sex', 'tissue', etc.
            train_attribute = getattr(self.config.data.train_test_split.train, method, "unknown")
            test_attribute = getattr(self.config.data.train_test_split.test, method, "unknown")

            # Capitalize the first letter of train and test attributes
            train_attribute = train_attribute.capitalize()
            test_attribute = test_attribute.capitalize()

            # Format the file name based on the template using the method-specific attributes
            file_name = file_name_template.format(
                f"train{train_attribute}", f"test{test_attribute}"
            )

            # Define the output file path
            output_file_path = os.path.join(self.output_dir, file_name)

            # Save DataFrame to CSV
            df_predictions.to_csv(output_file_path, index=False)

            logger.info(f"Predictions saved to {output_file_path}")

    def compute_metrics(self):
        """
        Run the metrics evaluation pipeline.
        """
        self._evaluate_model_performance()
        self._calculate_and_save_metrics()
        self.save_predictions_to_csv()
