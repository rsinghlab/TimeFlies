import contextlib
import json
import os
import sys

import dill as pickle
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from ..utils.path_manager import PathManager

# Aggressive TensorFlow logging suppression
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_DISABLE_MKL"] = "1"


# Suppress stderr during imports
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Import TensorFlow and related modules with suppressed stderr
with suppress_stderr():
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping


class CustomModelCheckpoint(tf.keras.callbacks.ModelCheckpoint):
    """
    CustomModelCheckpoint is a custom callback for saving model checkpoints.

    It inherits from the tf.keras.callbacks.ModelCheckpoint class and overrides some of its methods
    for saving model weights. In addition to the normal functionality, it also saves the best validation
    loss to a separate file and saves the model history to a separate file. This allows it to compare between an
    already saved model and a new model during training and save the new model only if it has a better validation loss.
    """

    def __init__(
        self,
        filepath,
        best_val_loss_path,
        label_path,
        label_encoder,
        reference_path,
        reference,
        scaler,
        scaler_path,
        is_scaler_fit,
        is_scaler_fit_path,
        highly_variable_genes,
        highly_variable_genes_path,
        mix_included,
        mix_included_path,
        num_features,
        num_features_path,
        metadata_path,
        path_manager=None,
        *args,
        **kwargs,
    ):
        """
        Initialize a CustomModelCheckpoint instance.

        Args:
            filepath (str): Path for saving the model weights.
            best_val_loss_path (str): Path for saving the best validation loss.
            label_path (str): Path for saving the label encoder.
            label_encoder (sklearn.preprocessing.LabelEncoder): The label encoder.
            reference_path (str): Path for saving the reference data.
            reference (numpy.ndarray): The reference data.
            scaler (sklearn.preprocessing.StandardScaler): The scaler.
            scaler_path (str): Path for saving the scaler.
            is_scaler_fit (bool): Whether the scaler has been fit or not.
            is_scaler_fit_path (str): Path for saving the is_scaler_fit variable.
            highly_variable_genes (list): List of highly variable genes.
            highly_variable_genes_path (str): Path for saving the highly variable genes.
            mix_included (bool): Whether mix is included.
            mix_included_path (str): Path for saving the mix_included variable.
            num_features (int): Number of features in the training data.
            num_features_path (str): Path to save the num_features variable.
            metadata_path (str): Path to the experiment metadata.json file.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """

        # Call the parent class's constructor
        super().__init__(filepath, *args, **kwargs)

        # Initialize arguments
        self.best_val_loss = float("inf")
        self.best_val_loss_path = best_val_loss_path

        self.label_path = label_path
        self.label_encoder = label_encoder

        self.reference_path = reference_path
        self.reference = reference

        self.scaler = scaler
        self.scaler_path = scaler_path

        self.is_scaler_fit = is_scaler_fit
        self.is_scaler_fit_path = is_scaler_fit_path

        self.mix_included = mix_included
        self.mix_included_path = mix_included_path

        self.highly_variable_genes = highly_variable_genes
        self.highly_variable_genes_path = highly_variable_genes_path

        self.num_features = num_features
        self.num_features_path = num_features_path
        
        self.metadata_path = metadata_path

        # Track initial and current best validation losses
        self.initial_best_val_loss = float("inf")  # The historical best before training
        self.model_improved = False  # True only if final model beats historical best
        
        # Store path_manager for models/ folder saving
        self.path_manager = path_manager

    def set_best_val_loss(self, best_val_loss):
        """
        Set the best validation loss.

        Args:
            best_val_loss (float): The best validation loss.
        """

        # Store the historical best before training starts
        self.initial_best_val_loss = best_val_loss
        # Set the current best validation loss
        self.best_val_loss = best_val_loss
        self.best = best_val_loss  # Update parent class's best variable

    def on_epoch_end(self, epoch, logs=None):
        """
        Method called at the end of an epoch during model's training. It checks if the current validation
        loss is better than the best validation loss seen so far and if so, saves the new best validation
        loss and calls the parent class's on_epoch_end method to save the model weights.

        Args:
            epoch (int): The number of the epoch that just finished.
            logs (dict, optional): Dictionary of logs, contains the metrics results for this training epoch.
        """

        # Get the current validation loss
        current_val_loss = logs.get("val_loss")

        # If the current validation loss is better than the best validation loss seen so far
        if current_val_loss < self.best_val_loss:
            self.best_val_loss = current_val_loss

            # Only set model_improved if this beats the historical best from before training
            if current_val_loss < self.initial_best_val_loss:
                self.model_improved = True

            # Custom clean message instead of verbose Keras output
            print(
                f"\nEpoch {epoch + 1}: val_loss improved from {self.best:.5f} to {current_val_loss:.5f}"
            )

            # Save best validation loss to a file
            with open(self.best_val_loss_path, "w") as f:
                json.dump({"best_val_loss": self.best_val_loss}, f)

            # Save the label encoder (if exists - None for regression)
            with open(self.label_path, "wb") as label_file:
                pickle.dump(
                    self.label_encoder, label_file
                )  # save the label_encoder when the model improves (None for regression)

            # Save reference data
            with open(self.reference_path, "wb") as reference_file:
                np.save(reference_file, self.reference)

            # Save scaler
            with open(self.scaler_path, "wb") as scaler_file:
                pickle.dump(self.scaler, scaler_file)

            # Save is_scaler_fit
            with open(self.is_scaler_fit_path, "wb") as is_scaler_fit_file:
                pickle.dump(self.is_scaler_fit, is_scaler_fit_file)

            # Save highly variable genes
            with open(
                self.highly_variable_genes_path, "wb"
            ) as highly_variable_genes_file:
                pickle.dump(self.highly_variable_genes, highly_variable_genes_file)

            with open(self.mix_included_path, "wb") as mix_included_file:
                pickle.dump(self.mix_included, mix_included_file)

            # Save num_features
            with open(self.num_features_path, "wb") as f:
                pickle.dump(self.num_features, f)

            # Call the parent class's on_epoch_end method to save the model weights
            # Temporarily suppress parent's verbose output
            original_verbose = self.verbose
            self.verbose = 0
            super().on_epoch_end(epoch, logs)
            self.verbose = original_verbose
            
            # Also save to models/ folder for reuse across evaluations
            if self.path_manager and self.model_improved:
                self._save_to_models_folder()
    
    def _save_to_models_folder(self):
        """Save model artifacts to models/ folder for reuse across evaluations."""
        import shutil
        from pathlib import Path
        
        # Get models folder path
        models_dir = Path(self.path_manager.get_models_folder_path())
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all model artifacts to models/ folder
        
        artifacts = [
            (self.filepath, models_dir / Path(self.filepath).name),
            (self.best_val_loss_path, models_dir / "best_val_loss.json"),
            (self.label_path, models_dir / "label_encoder.pkl"),
            (self.reference_path, models_dir / "reference_data.npy"),
            (self.scaler_path, models_dir / "scaler.pkl"),
            (self.is_scaler_fit_path, models_dir / "is_scaler_fit.pkl"),
            (self.highly_variable_genes_path, models_dir / "highly_variable_genes.pkl"),
            (self.mix_included_path, models_dir / "mix_included.pkl"),
            (self.num_features_path, models_dir / "num_features.pkl"),
        ]
        
        for source, dest in artifacts:
            if Path(source).exists():
                shutil.copy2(source, dest)


class ModelLoader:
    """
    A class to manage the loading of machine learning or deep learning models.

    This class constructs the full path to the saved model based on the configuration settings
    and loads the model along with associated components, such as label encoders and other
    necessary preprocessing objects, to ensure the model is ready for inference.

    Attributes:
    - config (ConfigHandler): Holds configuration settings for loading the model and related components.
    - model_dir (str): Directory path where the model files are stored, constructed from configuration settings.
    - model_path (str): Full path to the specific model file to be loaded.
    - model_type (str): The type of model to be loaded (e.g., "cnn", "rnn"), specified in the configuration.
    """

    def __init__(
        self,
        config,
        pipeline_mode="training",
        use_models_folder=None,
    ):
        """
        Initializes the ModelLoader with configuration and directory structure to locate the model files.

        Parameters:
        - config (ConfigHandler): A ConfigHandler instance containing settings for model loading, paths,
          and preprocessing components.
        - pipeline_mode (str): "training" (train+eval) or "evaluation" (eval-only)
        - use_models_folder (bool): If True, load from models/ folder; if False, use best experiment;
                                   if None, auto-detect based on pipeline_mode

        Sets up:
        - `model_dir` by constructing the directory path from config details.
        - `model_path` as the specific file path for the saved model.
        - `model_type` to specify the type of model (e.g., CNN, RNN) as per config.
        """
        self.config = config
        self.pipeline_mode = pipeline_mode
        self.path_manager = PathManager(self.config)
        self.model_type = getattr(self.config.data, "model", "CNN").lower()
        
        # Determine whether to use models/ folder or best experiment
        if use_models_folder is None:
            # Auto-detect: use models/ for evaluation, best experiment for training
            use_models_folder = (pipeline_mode == "evaluation")
        
        if use_models_folder:
            # Use models/ folder for trained model (evaluation mode)
            self.model_dir = self.path_manager.get_models_folder_path()
        else:
            # Use best experiment for current config (training mode) 
            self.model_dir = self.path_manager.get_best_model_dir_for_config()
            
        self.model_path = self._get_model_path()

    def _get_model_path(self):
        """
        Determines the model file path based on the model type.

        Returns:
        - str: The path to the model file.
        """
        if self.model_type in ["cnn", "mlp"]:
            model_filename = "model.keras"
        else:
            model_filename = "model.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        return model_path

    def _verify_split_compatibility(self):
        """
        Verify that the current config's split settings are compatible with the saved model.
        Raises a warning or error if there's a mismatch that could cause issues.
        """
        import json

        from ..utils.split_naming import SplitNamingUtils

        # Look for metadata.json in the model directory
        metadata_path = os.path.join(self.model_dir, "metadata.json")

        if not os.path.exists(metadata_path):
            print(
                "WARNING: No metadata found for saved model, cannot verify split compatibility"
            )
            return

        try:
            with open(metadata_path) as f:
                saved_metadata = json.load(f)

            # Extract current configuration
            current_split = SplitNamingUtils.extract_split_details_for_metadata(
                self.config
            )

            # Check all key metadata fields for compatibility
            mismatches = []

            # Check model type
            current_model = getattr(self.config.model, "model_type", "CNN").upper()
            saved_model = saved_metadata.get("model_type", "")
            if current_model != saved_model:
                mismatches.append(f"Model type: {current_model} vs {saved_model}")

            # Check target variable
            current_target = getattr(self.config.data, "target_variable", "age")
            saved_target = saved_metadata.get("target", "")
            if current_target != saved_target:
                mismatches.append(f"Target: {current_target} vs {saved_target}")

            # Check tissue
            current_tissue = getattr(self.config.data, "tissue", "head")
            saved_tissue = saved_metadata.get("tissue", "")
            if current_tissue != saved_tissue:
                mismatches.append(f"Tissue: {current_tissue} vs {saved_tissue}")

            # Check batch correction
            current_batch = getattr(self.config.data.batch_correction, "enabled", False)
            saved_batch = saved_metadata.get("batch_correction", None)
            if saved_batch is not None and current_batch != saved_batch:
                mismatches.append(f"Batch correction: {current_batch} vs {saved_batch}")

            # Check split configuration
            saved_split = saved_metadata.get("split_config", {})
            if not saved_split:
                mismatches.append("Split configuration missing in saved model")
            else:
                method_match = current_split.get("method") == saved_split.get("method")
                split_name_match = current_split.get("split_name") == saved_split.get(
                    "split_name"
                )

                if not method_match:
                    mismatches.append(
                        f"Split method: {current_split.get('method')} vs {saved_split.get('method')}"
                    )
                if not split_name_match:
                    mismatches.append(
                        f"Split name: {current_split.get('split_name')} vs {saved_split.get('split_name')}"
                    )

            # Report results
            if mismatches:
                print("WARNING: Configuration mismatches detected with saved model!")
                for mismatch in mismatches:
                    print(f"  - {mismatch}")
                print(
                    "  This may cause evaluation issues if configuration differs from model training"
                )
            elif self.pipeline_mode == "evaluation":
                # Only show success message during evaluation-only pipeline
                print("✓ Configuration matches saved model")

        except Exception as e:
            print(f"WARNING: Could not verify split compatibility: {e}")

    def load_model(self):
        """
        Loads the saved model and related components.

        This method constructs the file paths for the model and other related files
        (like label encoder, scaler, etc.) based on the configuration settings.
        It then loads these components and returns them for use.

        Returns:
        - tuple: A tuple containing the loaded model and related components like label encoder, reference data, scaler, test data, test labels, and training history.
        """
        # Verify split compatibility before loading
        self._verify_split_compatibility()

        # Load the model
        if os.path.exists(self.model_path):
            if self.model_type in ["cnn", "mlp"]:
                # Suppress the compile warning for loaded models
                import logging

                # Temporarily suppress absl warnings
                absl_logger = logging.getLogger("absl")
                old_level = absl_logger.level
                absl_logger.setLevel(logging.ERROR)

                model = tf.keras.models.load_model(self.model_path)

                # Restore logging level
                absl_logger.setLevel(old_level)
            else:
                model = self._load_pickle(self.model_path)
        else:
            print(f"ERROR: Model file not found: {self.model_path}")
            sys.exit(1)

        # Return all loaded components
        return model

    def load_model_components(self):
        """
        Loads the saved model's related components.

        This method constructs the file paths for the model and other related files
        (like label encoder, scaler, etc.) based on the configuration settings.
        It then loads these components and returns them for use.

        Returns:
        - tuple: A tuple containing the loaded model and related components like label encoder, reference data, scaler, test data, test labels, and training history.
        """

        # Load other related components from the model directory
        label_encoder = self._load_component_file("label_encoder.pkl")
        scaler = self._load_component_file("scaler.pkl")
        is_scaler_fit = self._load_component_file("is_scaler_fit.pkl")
        highly_variable_genes = self._load_component_file("highly_variable_genes.pkl")
        num_features = self._load_component_file("num_features.pkl")
        mix_included = self._load_component_file("mix_included.pkl")

        reference_data = self._load_component_file(
            "reference_data.npy", file_type="numpy"
        )

        # History is in training subdirectory
        history = self._load_training_file("history.pkl")

        # Return all loaded components
        return (
            label_encoder,
            scaler,
            is_scaler_fit,
            highly_variable_genes,
            num_features,
            history,
            mix_included,
            reference_data,
        )

    def _load_pickle(self, file_path):
        """
        Loads a pickle file from the given path.

        Parameters:
        - file_path (str): The path to the pickle file.

        Returns:
        - object: The object loaded from the pickle file.
        """
        with open(file_path, "rb") as file:
            return pickle.load(file)

    def _load_file(self, file_name, file_type="pickle"):
        """
        Loads a file from the model directory based on the file type.

        Parameters:
        - file_name (str): The name of the file to be loaded.
        - file_type (str): The type of the file to be loaded, either 'pickle' or 'numpy'.

        Returns:
        - object: The object loaded from the file, depending on its type (pickle or numpy).
        """
        file_path = os.path.join(self.model_dir, file_name)

        if os.path.exists(file_path):
            if file_type == "pickle":
                return self._load_pickle(file_path)
            elif file_type == "numpy":
                return np.load(file_path, allow_pickle=True)
        else:
            # Print error if the file does not exist and exit the program
            print(f"Error: {file_name} not found in {file_path}")
            sys.exit(1)

    def _load_component_file(self, file_name, file_type="pickle"):
        """
        Loads a component file, checking both new model_components/ and old root directory.

        Parameters:
        - file_name (str): The name of the file to be loaded.
        - file_type (str): The type of the file to be loaded, either 'pickle' or 'numpy'.

        Returns:
        - object: The object loaded from the file.
        """
        # Try new model_components directory first
        components_dir = os.path.join(self.model_dir, "model_components")
        new_path = os.path.join(components_dir, file_name)

        if os.path.exists(new_path):
            if file_type == "pickle":
                return self._load_pickle(new_path)
            elif file_type == "numpy":
                return np.load(new_path, allow_pickle=True)

        # Fallback to old location (root of model directory)
        old_path = os.path.join(self.model_dir, file_name)
        if os.path.exists(old_path):
            if file_type == "pickle":
                return self._load_pickle(old_path)
            elif file_type == "numpy":
                return np.load(old_path, allow_pickle=True)

        # File not found in either location
        print(f"Error: {file_name} not found in {new_path} or {old_path}")
        sys.exit(1)

    def _load_training_file(self, file_name, file_type="pickle"):
        """
        Loads a training file from models/ folder first, then fallback to training/ subdirectory or root.

        Parameters:
        - file_name (str): The name of the file to be loaded.
        - file_type (str): The type of the file to be loaded.

        Returns:
        - object: The object loaded from the file.
        """
        # For history.pkl, try models/ folder first (new location)
        if file_name == "history.pkl":
            models_path = os.path.join(self.path_manager.get_models_folder_path(), file_name)
            if os.path.exists(models_path):
                if file_type == "pickle":
                    return self._load_pickle(models_path)
                elif file_type == "numpy":
                    return np.load(models_path, allow_pickle=True)

        # Try training directory (old location for history)
        training_dir = os.path.join(self.model_dir, "training")
        new_path = os.path.join(training_dir, file_name)

        if os.path.exists(new_path):
            if file_type == "pickle":
                return self._load_pickle(new_path)
            elif file_type == "numpy":
                return np.load(new_path, allow_pickle=True)

        # Fallback to old location (root of model directory)
        old_path = os.path.join(self.model_dir, file_name)
        if os.path.exists(old_path):
            if file_type == "pickle":
                return self._load_pickle(old_path)
            elif file_type == "numpy":
                return np.load(old_path, allow_pickle=True)

        # File not found in any location
        if file_name == "history.pkl":
            print(f"Error: {file_name} not found in {self.path_manager.get_models_folder_path()}, {new_path} or {old_path}")
        else:
            print(f"Error: {file_name} not found in {new_path} or {old_path}")
        sys.exit(1)


class ModelBuilder:
    """
    A class to handle model building and training.

    This class constructs and trains a model based on the provided training data
    and configuration settings.
    """

    def __init__(
        self,
        config,
        train_data,
        train_labels,
        label_encoder,
        reference_data,
        scaler,
        is_scaler_fit,
        highly_variable_genes,
        mix_included,
        experiment_name=None,
    ):
        """
        Initializes the ModelBuilder with the given configuration and training data.

        Parameters:
        - config (ConfigHandler): A ConfigHandler object containing configuration settings.
        - train_data (numpy.ndarray): The training data.
        - train_labels (numpy.ndarray): The labels for the training data.
        - label_encoder (LabelEncoder): The label encoder for encoding labels.
        - reference_data (numpy.ndarray): Reference data used in model training.
        - scaler (object): Scaler object used for feature scaling.
        - is_scaler_fit (bool): Flag indicating if the scaler has been fitted.
        - highly_variable_genes (list): List of highly variable genes used in training.
        - mix_included (bool): Flag indicating if mix_included feature is used.
        """
        self.config = config
        self.train_data = train_data
        self.train_labels = train_labels
        self.label_encoder = label_encoder
        self.reference_data = reference_data
        self.scaler = scaler
        self.is_scaler_fit = is_scaler_fit
        self.highly_variable_genes = highly_variable_genes
        self.mix_included = mix_included
        self.experiment_name = experiment_name
        self.model_type = getattr(self.config.data, "model", "CNN").lower()

    def create_cnn_model(self, num_output_units):
        """
        Create a Convolutional Neural Network (CNN) model using TensorFlow and the provided configuration.

        Args:
            num_output_units (int): The number of output units for the final layer of the model.

        Returns:
            model (tensorflow.python.keras.Model): The created and compiled CNN model.
        """
        cnn_config = getattr(self.config.model, "cnn", {})

        # Create model
        model = tf.keras.Sequential()

        # Convolutional blocks
        for i in range(len(cnn_config.filters)):
            if i == 0:
                # First layer needs input_shape
                model.add(
                    tf.keras.layers.Conv1D(
                        filters=cnn_config.filters[i],
                        kernel_size=cnn_config.kernel_sizes[i],
                        strides=cnn_config.strides[i],
                        padding=cnn_config.paddings[i],
                        input_shape=(1, self.train_data.shape[2]),
                    )
                )
            else:
                # Subsequent layers don't need input_shape
                model.add(
                    tf.keras.layers.Conv1D(
                        filters=cnn_config.filters[i],
                        kernel_size=cnn_config.kernel_sizes[i],
                        strides=cnn_config.strides[i],
                        padding=cnn_config.paddings[i],
                    )
                )
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.ReLU())
            if cnn_config.pool_sizes[i] is not None:
                model.add(
                    tf.keras.layers.MaxPooling1D(
                        pool_size=cnn_config.pool_sizes[i],
                        strides=cnn_config.pool_strides[i],
                        padding="same",
                    )
                )

        # Fully connected layers
        model.add(tf.keras.layers.Flatten())
        for units in cnn_config.dense_units:
            model.add(
                tf.keras.layers.Dense(units=units, activation=cnn_config.activation)
            )
            model.add(tf.keras.layers.Dropout(rate=cnn_config.dropout_rate))
        # Output layer based on task type
        task_type = getattr(self.config.model, "task_type", "classification")
        if task_type == "regression":
            model.add(tf.keras.layers.Dense(units=1, activation="linear"))
            default_loss = "mse"
            default_metrics = ["mae"]
        else:
            model.add(
                tf.keras.layers.Dense(units=num_output_units, activation="softmax")
            )
            default_loss = "categorical_crossentropy"
            default_metrics = ["accuracy", "auc"]

        # Use standard Adam optimizer for all platforms (Keras 3 compatible)
        learning_rate = getattr(self.config.model.training, "learning_rate", 0.001)
        optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Get loss from config
        cnn_config = getattr(self.config.model, "cnn", {})
        loss = getattr(cnn_config, "loss", default_loss)

        # Get metrics from config
        eval_config = getattr(self.config, "evaluation", {})
        config_metrics = eval_config.get("metrics", {})
        training_metrics = config_metrics.get("training", {}).get(
            task_type, default_metrics
        )

        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss=loss,
            metrics=training_metrics,
        )

        return model

    def create_mlp_model(self, num_output_units):
        """
        Create a Multilayer Perceptron (MLP) model using TensorFlow and the provided configuration.

        Args:
            num_output_units (int): The number of output units for the final layer of the model.

        Returns:
            model (tensorflow.python.keras.Model): The created and compiled MLP model.
        """
        mlp_config = getattr(self.config.model, "mlp", {})

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=(self.train_data.shape[1],)))

        # Fully connected layers
        for units in mlp_config.units:
            model.add(
                tf.keras.layers.Dense(
                    units=units, activation=mlp_config.activation_function
                )
            )
            model.add(tf.keras.layers.Dropout(rate=mlp_config.dropout_rate))

        # Output layer based on task type
        task_type = getattr(self.config.model, "task_type", "classification")
        if task_type == "regression":
            model.add(tf.keras.layers.Dense(units=1, activation="linear"))
            default_loss = "mse"
            default_metrics = ["mae"]
        else:
            model.add(
                tf.keras.layers.Dense(units=num_output_units, activation="softmax")
            )
            default_loss = "categorical_crossentropy"
            default_metrics = ["accuracy", "auc"]

        # Use standard Adam optimizer for all platforms (Keras 3 compatible)
        optimizer_instance = tf.keras.optimizers.Adam(
            learning_rate=mlp_config.learning_rate
        )

        # Get loss from config
        loss = getattr(mlp_config, "loss", default_loss)

        # Get metrics from config
        eval_config = getattr(self.config, "evaluation", {})
        config_metrics = eval_config.get("metrics", {})
        training_metrics = config_metrics.get("training", {}).get(
            task_type, default_metrics
        )

        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss=loss,
            metrics=training_metrics,
        )

        return model

    def create_logistic_regression(self):
        """
        Create a regression model using scikit-learn and the provided configuration.

        Returns:
            lr: The regression model (linear or logistic based on task type).
        """
        lr_config = getattr(
            self.config.model, "logistic", {}
        )  # Note: using "logistic" to match config
        task_type = getattr(self.config.model, "task_type", "classification")

        if task_type == "regression":
            from sklearn.linear_model import LinearRegression

            lr = LinearRegression()
        else:
            lr = LogisticRegression(
                penalty=getattr(lr_config, "penalty", "l2"),
                solver=getattr(lr_config, "solver", "lbfgs"),
                max_iter=getattr(lr_config, "max_iter", 1000),
                C=getattr(lr_config, "C", 1.0),
                random_state=getattr(lr_config, "random_state", 42),
            )

        return lr

    def create_random_forest(self):
        """
        Create a random forest model using scikit-learn and the provided configuration.

        Returns:
            rf: The random forest model (classifier or regressor based on task type).
        """
        rf_config = getattr(self.config.model, "random_forest", {})
        task_type = getattr(self.config.model, "task_type", "classification")

        if task_type == "regression":
            from sklearn.ensemble import RandomForestRegressor

            rf = RandomForestRegressor(
                n_estimators=rf_config.n_estimators,
                max_depth=rf_config.max_depth,
                min_samples_split=rf_config.min_samples_split,
                min_samples_leaf=rf_config.min_samples_leaf,
                max_features=rf_config.max_features,
                bootstrap=rf_config.bootstrap,
                oob_score=rf_config.oob_score,
                n_jobs=rf_config.n_jobs,
                random_state=rf_config.random_state,
            )
        else:
            rf = RandomForestClassifier(
                n_estimators=rf_config.n_estimators,
                criterion=rf_config.criterion,
                max_depth=rf_config.max_depth,
                min_samples_split=rf_config.min_samples_split,
                min_samples_leaf=rf_config.min_samples_leaf,
                max_features=rf_config.max_features,
                bootstrap=rf_config.bootstrap,
                oob_score=rf_config.oob_score,
                n_jobs=rf_config.n_jobs,
                random_state=rf_config.random_state,
            )

        return rf

    def create_xgboost_model(self):
        """
        Create an XGBoost classifier using xgboost and the provided configuration.

        Returns:
            model (xgboost.XGBClassifier): The XGBoost classifier.
        """
        xgb_config = getattr(self.config.model, "xgboost", {})

        # Determine task type and set corresponding XGBoost parameters
        task_type = getattr(self.config.model, "task_type", "classification")
        if task_type == "regression":
            xgb_objective = "reg:squarederror"
            eval_metric = getattr(xgb_config, "eval_metric", "rmse")
        else:
            # Classification - determine if binary or multiclass based on number of classes
            num_classes = (
                len(np.unique(self.train_labels))
                if hasattr(self, "train_labels")
                else 2
            )
            if num_classes == 2:
                xgb_objective = "binary:logistic"
                eval_metric = getattr(xgb_config, "eval_metric", "auc")
            else:
                xgb_objective = "multi:softmax"
                eval_metric = getattr(xgb_config, "eval_metric", "mlogloss")

        # Basic XGBoost parameters
        xgb_params = {
            "objective": xgb_objective,
            "eval_metric": eval_metric,
            "learning_rate": xgb_config.learning_rate,
            "n_estimators": xgb_config.n_estimators,
            "max_depth": xgb_config.max_depth,
            "min_child_weight": xgb_config.min_child_weight,
            "subsample": xgb_config.subsample,
            "colsample_bytree": xgb_config.colsample_bytree,
            "random_state": xgb_config.random_state,
            "tree_method": xgb_config.tree_method,
            "predictor": xgb_config.predictor,
        }

        # Initialize XGBoost model based on task type
        if task_type == "regression":
            model = xgb.XGBRegressor(**xgb_params)
        else:
            # Adjust parameters for multiclass classification
            if xgb_objective == "multi:softmax":
                xgb_params["num_class"] = len(np.unique(self.train_labels))
            model = xgb.XGBClassifier(**xgb_params)

        model.set_params(early_stopping_rounds=xgb_config.early_stopping_rounds)

        return model

    def build_model(self):
        """
        Build a model based on the specified type in the config.

        Returns:
            model: The created model, which could be CNN, MLP, or logistic regression.
        """
        num_output_units = (
            self.train_labels.shape[1] if self.model_type in ["cnn", "mlp"] else None
        )

        if self.model_type == "cnn":
            model = self.create_cnn_model(num_output_units)
        elif self.model_type == "mlp":
            model = self.create_mlp_model(num_output_units)
        elif self.model_type == "logisticregression":
            model = self.create_logistic_regression()
        elif self.model_type == "randomforest":
            model = self.create_random_forest()
        elif self.model_type == "xgboost":
            model = self.create_xgboost_model()
        else:
            raise ValueError("Unsupported model type provided.")
        return model

    def train_model(self, model):
        """
        Train a model and save it if it's the best one based on validation accuracy.
        """
        # Prepare directories and paths
        model_dir = self._prepare_directories()
        paths = self._define_paths(model_dir)

        if self.model_type in ["cnn", "mlp"]:
            history, model_improved = self._train_neural_network(model, paths)
        else:
            history, model_improved = self._train_sklearn_model(model, paths)

        return history, model, model_improved

    def _prepare_directories(self):
        """
        Prepare directories for saving models and related artifacts in experiment structure.

        Returns:
            experiment_dir (str): The directory where the experiment and artifacts will be saved.
        """
        self.path_manager = PathManager(self.config)

        # Use experiment directory instead of old model directory
        if self.experiment_name:
            experiment_dir = self.path_manager.get_experiment_dir(self.experiment_name)
        else:
            experiment_dir = self.path_manager.get_experiment_dir()

        os.makedirs(experiment_dir, exist_ok=True)
        return experiment_dir

    def _define_paths(self, experiment_dir):
        """
        Define paths for saving model checkpoints and related artifacts in experiment structure.

        Args:
            experiment_dir (str): The experiment directory where everything will be saved.

        Returns:
            dict: A dictionary containing paths for various artifacts.
        """
        # Create model_components subdirectory for cleaner organization
        components_dir = os.path.join(experiment_dir, "model_components")
        training_dir = os.path.join(experiment_dir, "training")
        os.makedirs(components_dir, exist_ok=True)
        os.makedirs(training_dir, exist_ok=True)

        paths = {
            "label_path": os.path.join(components_dir, "label_encoder.pkl"),
            "reference_path": os.path.join(components_dir, "reference_data.npy"),
            "scaler_path": os.path.join(components_dir, "scaler.pkl"),
            "is_scaler_fit_path": os.path.join(components_dir, "is_scaler_fit.pkl"),
            "num_features_path": os.path.join(components_dir, "num_features.pkl"),
            "highly_variable_genes_path": os.path.join(
                components_dir, "highly_variable_genes.pkl"
            ),
            "mix_included_path": os.path.join(components_dir, "mix_included.pkl"),
            "history_path": os.path.join(training_dir, "history.pkl"),
            "metadata_path": os.path.join(experiment_dir, "metadata.json"),
            "experiment_dir": experiment_dir,
            "components_dir": components_dir,
        }
        return paths

    def _train_neural_network(self, model, paths):
        """
        Train a neural network model (CNN or MLP) and save the best model based on validation loss.

        Args:
            model: The neural network model to train.
            paths (dict): A dictionary containing paths for saving artifacts.

        Returns:
            history: The training history.
        """
        custom_model_path = os.path.join(paths["experiment_dir"], "model.keras")
        # Try to load best validation loss from best symlink first, then current experiment
        from pathlib import Path

        # Get the correct best symlink path: base/best/config_key/model_components/best_val_loss.json
        config_key = self.path_manager.get_config_key()
        # Get base path and manually construct the correct best path
        base_path = Path(self.path_manager._get_project_root()) / "outputs"
        project_name = getattr(self.path_manager.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.path_manager.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        task_type = getattr(
            self.path_manager.config.model, "task_type", "classification"
        )

        # best_symlink_path variable removed - was unused
        current_path = os.path.join(paths["components_dir"], "best_val_loss.json")

        self.num_features = (
            self.train_data.shape[2]
            if self.model_type == "cnn"
            else self.train_data.shape[1]
        )

        # Load the best validation loss from all experiments
        best_val_loss = float("inf")

        # First, scan all experiments in all_runs to find the true historical best
        # This is more reliable than relying on potentially broken symlinks
        try:
            all_runs_path = str(
                base_path
                / project_name
                / correction_dir
                / task_type
                / "experiments"
                / "all_runs"
                / config_key
            )
            if os.path.exists(all_runs_path) and os.path.isdir(all_runs_path):
                for experiment_dir in sorted(os.listdir(all_runs_path)):
                    if experiment_dir.startswith("experiment_"):
                        exp_best_val_path = os.path.join(
                            all_runs_path,
                            experiment_dir,
                            "model_components",
                            "best_val_loss.json",
                        )
                        # Check if file exists and is not a broken symlink
                        if os.path.exists(exp_best_val_path) and os.path.isfile(
                            exp_best_val_path
                        ):
                            try:
                                with open(exp_best_val_path) as f:
                                    exp_val_loss = json.load(f)["best_val_loss"]
                                    if exp_val_loss < best_val_loss:
                                        best_val_loss = exp_val_loss
                            except (FileNotFoundError, json.JSONDecodeError, OSError):
                                continue
        except (OSError, Exception):
            # Silently continue if we can't scan the all_runs directory
            pass

        # If we still don't have a best val loss, try the current path (but not broken symlinks)
        if best_val_loss == float("inf") and current_path:
            try:
                if os.path.exists(current_path) and os.path.isfile(current_path):
                    with open(current_path) as f:
                        loaded_val_loss = json.load(f)["best_val_loss"]
                        if loaded_val_loss < best_val_loss:
                            best_val_loss = loaded_val_loss
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

        # Previous model info will be shown in TRAINING PROGRESS header

        # Split data into training and validation sets
        (
            train_inputs_split,
            val_inputs_split,
            train_labels_split,
            val_labels_split,
        ) = train_test_split(
            self.train_data,
            self.train_labels,
            test_size=getattr(self.config.model.training, "validation_split", 0.2),
            random_state=getattr(self.config.general, "random_state", 42),
            stratify=self.train_labels,
        )

        # Define callbacks for early stopping and model saving
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=getattr(self.config.model.training, "early_stopping_patience", 10),
        )
        model_checkpoint = CustomModelCheckpoint(
            custom_model_path,
            current_path,
            paths["label_path"],
            self.label_encoder,
            paths["reference_path"],
            self.reference_data,
            self.scaler,
            paths["scaler_path"],
            self.is_scaler_fit,
            paths["is_scaler_fit_path"],
            self.highly_variable_genes,
            paths["highly_variable_genes_path"],
            self.mix_included,
            paths["mix_included_path"],
            self.num_features,
            paths["num_features_path"],
            paths["metadata_path"],
            path_manager=PathManager(self.config),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,  # Suppress checkpoint messages
        )
        model_checkpoint.set_best_val_loss(best_val_loss)

        # Fit the model with validation split
        history = model.fit(
            train_inputs_split,
            train_labels_split,
            epochs=getattr(self.config.model.training, "epochs", 100),
            batch_size=getattr(self.config.model.training, "batch_size", 32),
            validation_data=(val_inputs_split, val_labels_split),
            callbacks=[early_stopping, model_checkpoint],
            verbose=1,  # Show progress bar per epoch
        )

        # Save history to models/ folder if model improved
        if model_checkpoint.model_improved:
            models_dir = PathManager(self.config).get_models_folder_path()
            os.makedirs(models_dir, exist_ok=True)
            
            # Save history
            history_path = os.path.join(models_dir, "history.pkl")
            with open(history_path, "wb") as f:
                pickle.dump(history.history, f)
                

        return history, model_checkpoint.model_improved

    def _train_sklearn_model(self, model, paths):
        """
        Train a scikit-learn model (Logistic Regression, Random Forest, or XGBoost) and save the best model based on validation accuracy.

        Args:
            model: The scikit-learn model to train.
            paths (dict): A dictionary containing paths for saving artifacts.

        Returns:
            history: The training history (if applicable).
        """
        train_labels = np.argmax(self.train_labels, axis=1)
        # Calculate the validation split index
        (
            train_inputs_split,
            val_inputs_split,
            train_labels_split,
            val_labels_split,
        ) = train_test_split(
            self.train_data,
            train_labels,
            test_size=getattr(self.config.model.training, "validation_split", 0.2),
            random_state=getattr(self.config.general, "random_state", 42),
            stratify=train_labels,
        )

        if self.model_type in ["logisticregression", "randomforest"]:
            model.fit(train_inputs_split, train_labels_split)
            history = None
        elif self.model_type == "xgboost":
            # Training with evaluation set
            eval_set = [
                (train_inputs_split, train_labels_split),
                (val_inputs_split, val_labels_split),
            ]
            model.fit(
                train_inputs_split,
                train_labels_split,
                eval_set=eval_set,
                verbose=0,  # Suppress XGBoost training output
            )
            history = model.evals_result()
        else:
            raise ValueError("Unsupported model type provided.")

        # Evaluate the model
        val_predictions = model.predict(val_inputs_split)
        val_accuracy = accuracy_score(val_labels_split, val_predictions)

        # Load the best validation accuracy from best symlink first, then current experiment
        from pathlib import Path

        # Get the correct best symlink path: base/best/config_key/model_components/best_val_accuracy.json
        config_key = self.path_manager.get_config_key()
        # Get base path and manually construct the correct best path
        base_path = Path(self.path_manager._get_project_root()) / "outputs"
        project_name = getattr(self.path_manager.config, "project", "fruitfly_aging")
        batch_correction_enabled = getattr(
            self.path_manager.config.data.batch_correction, "enabled", False
        )
        correction_dir = (
            "batch_corrected" if batch_correction_enabled else "uncorrected"
        )

        task_type = getattr(
            self.path_manager.config.model, "task_type", "classification"
        )

        # best_symlink_accuracy_path variable removed - was unused
        current_accuracy_path = os.path.join(
            paths["components_dir"], "best_val_accuracy.json"
        )

        # Load the best validation accuracy from file
        historical_best_accuracy = 0

        # First, scan all experiments in all_runs to find the true historical best
        # This is more reliable than relying on potentially broken symlinks
        try:
            all_runs_path = str(
                base_path
                / project_name
                / correction_dir
                / task_type
                / "experiments"
                / "all_runs"
                / config_key
            )
            if os.path.exists(all_runs_path) and os.path.isdir(all_runs_path):
                for experiment_dir in sorted(os.listdir(all_runs_path)):
                    if experiment_dir.startswith("experiment_"):
                        exp_best_acc_path = os.path.join(
                            all_runs_path,
                            experiment_dir,
                            "model_components",
                            "best_val_accuracy.json",
                        )
                        # Check if file exists and is not a broken symlink
                        if os.path.exists(exp_best_acc_path) and os.path.isfile(
                            exp_best_acc_path
                        ):
                            try:
                                with open(exp_best_acc_path) as f:
                                    exp_val_acc = json.load(f)["best_val_accuracy"]
                                    if exp_val_acc > historical_best_accuracy:
                                        historical_best_accuracy = exp_val_acc
                            except (FileNotFoundError, json.JSONDecodeError, OSError):
                                continue
        except (OSError, Exception):
            # Silently continue if we can't scan the all_runs directory
            pass

        # If we still don't have a best accuracy, try the current path (but not broken symlinks)
        if historical_best_accuracy == 0 and current_accuracy_path:
            try:
                if os.path.exists(current_accuracy_path) and os.path.isfile(
                    current_accuracy_path
                ):
                    with open(current_accuracy_path) as f:
                        loaded_val_acc = json.load(f)["best_val_accuracy"]
                        if loaded_val_acc > historical_best_accuracy:
                            historical_best_accuracy = loaded_val_acc
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass

        # best_val_accuracy variable removed - was unused

        # Check if the current model outperforms the historical best (not just current best)
        print("Validation accuracy:", val_accuracy)
        print("Best validation accuracy so far:", historical_best_accuracy)
        model_improved = val_accuracy > historical_best_accuracy
        print("Model improved:", model_improved)

        if model_improved:
            # Update the record with the new best validation accuracy
            with open(current_accuracy_path, "w") as f:
                json.dump({"best_val_accuracy": val_accuracy}, f)

            # Save selected features for reference in the future
            num_features = self.train_data.shape[1]
            with open(paths["num_features_path"], "wb") as f:
                pickle.dump(num_features, f)

            # Save the model as it's the best one so far using pickle
            model_path = os.path.join(paths["experiment_dir"], "model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # XGBoost history is saved to models/ folder in pipeline_manager after training

            # Save the label encoder
            with open(paths["label_path"], "wb") as label_file:
                pickle.dump(
                    self.label_encoder, label_file
                )  # save the label_encoder when the model improves

            # Save reference data
            np.save(paths["reference_path"], self.reference_data)

            # Save scaler
            with open(paths["scaler_path"], "wb") as scaler_file:
                pickle.dump(self.scaler, scaler_file)

            # Save is_scaler_fit
            with open(paths["is_scaler_fit_path"], "wb") as is_scaler_fit_file:
                pickle.dump(self.is_scaler_fit, is_scaler_fit_file)

            with open(paths["mix_included_path"], "wb") as mix_included_file:
                pickle.dump(self.mix_included, mix_included_file)

            # Save highly variable genes
            with open(
                paths["highly_variable_genes_path"], "wb"
            ) as highly_variable_genes_file:
                pickle.dump(self.highly_variable_genes, highly_variable_genes_file)

            print("New best model saved with validation accuracy:", val_accuracy)

            # Save history to models/ folder for XGBoost (if it exists)
            if self.model_type == "xgboost" and history is not None:
                models_dir = PathManager(self.config).get_models_folder_path()
                os.makedirs(models_dir, exist_ok=True)
                history_path = os.path.join(models_dir, "history.pkl")
                with open(history_path, "wb") as f:
                    pickle.dump(history, f)

        return history, model_improved

    def run(self):
        """
        Builds and trains the model based on the configuration settings.

        This method constructs the model using the specified architecture (e.g., CNN, MLP),
        and then trains it using the training data.

        Returns:
        - tuple: A tuple containing the trained model and the training history.
        """
        # Build the model using the provided configuration
        model = self.build_model()

        # Train the model using the provided training data and additional components
        history, model, model_improved = self.train_model(model)

        return model, history, model_improved
