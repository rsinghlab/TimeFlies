import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import json
import dill as pickle
from utilities import PathManager
import numpy as np
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
        test_data,
        test_labels,
        test_data_path,
        test_labels_path,
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
            test_data (numpy.ndarray): Test data to save.
            test_labels (numpy.ndarray): Test labels to save.
            test_data_path (str): Path to save test data.
            test_labels_path (str): Path to save test labels.
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

        self.test_data = test_data
        self.test_labels = test_labels
        self.test_data_path = test_data_path
        self.test_labels_path = test_labels_path

        # Initialize a flag to indicate if the model improved during training
        self.model_improved = False

    def set_best_val_loss(self, best_val_loss):
        """
        Set the best validation loss.

        Args:
            best_val_loss (float): The best validation loss.
        """

        # Set the best validation loss
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
            self.model_improved = True

            # Save best validation loss to a file
            with open(self.best_val_loss_path, "w") as f:
                json.dump({"best_val_loss": self.best_val_loss}, f)

            # Save the label encoder
            with open(self.label_path, "wb") as label_file:
                pickle.dump(
                    self.label_encoder, label_file
                )  # save the label_encoder when the model improves

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

            # Save test data and labels
            with open(self.test_data_path, "wb") as test_data_file:
                np.save(test_data_file, self.test_data)

            with open(self.test_labels_path, "wb") as test_labels_file:
                np.save(test_labels_file, self.test_labels)

            # Call the parent class's on_epoch_end method to save the model weights
            super().on_epoch_end(epoch, logs)


class ModelLoader:
    """
    A class to handle model loading operations.

    This class constructs the path to the saved model based on the configuration
    and loads the model along with related components like label encoder and other preprocessing objects.
    """

    def __init__(
        self,
        config,
    ):
        """
        Initializes the ModelLoader with the given configuration and directory structure.

        Parameters:
        - config (ConfigHandler): A ConfigHandler object containing configuration settings.
        - code_dir (str): The directory where the code is located.
        """
        self.config = config
        self.path_manager = PathManager(self.config)

        # Use the shared method
        self.model_dir = self.path_manager.construct_model_directory()
        self.model_path = self._get_model_path()

    def _get_model_path(self):
        """
        Determines the model file path based on the model type.

        Returns:
        - str: The path to the model file.
        """
        model_type = self.config.DataParameters.GeneralSettings.model_type.lower()
        if model_type in ["cnn", "mlp"]:
            model_filename = "best_model.h5"
        else:
            model_filename = "best_model.pkl"
        model_path = os.path.join(self.model_dir, model_filename)
        return model_path

    def load_model(self):
        """
        Loads the saved model and related components.

        This method constructs the file paths for the model and other related files
        (like label encoder, scaler, etc.) based on the configuration settings.
        It then loads these components and returns them for use.

        Returns:
        - tuple: A tuple containing the loaded model and related components like label encoder, reference data, scaler, test data, test labels, and training history.
        """
        # Load the model
        if os.path.exists(self.model_path):
            model_type = self.config.DataParameters.GeneralSettings.model_type.lower()
            if model_type in ["cnn", "mlp"]:
                model = tf.keras.models.load_model(self.model_path)
            else:
                model = self._load_pickle(self.model_path)
        else:
            exit()

        # Load other related components from the model directory
        label_encoder = self._load_file("label_encoder.pkl")
        reference_data = self._load_file("reference_data.npy", file_type="numpy")
        scaler = self._load_file("scaler.pkl")
        is_scaler_fit = self._load_file("is_scaler_fit.pkl")
        mix_included = self._load_file("mix_included.pkl")
        highly_variable_genes = self._load_file("highly_variable_genes.pkl")
        test_data = self._load_file("test_data.npy", file_type="numpy")
        test_labels = self._load_file("test_labels.npy", file_type="numpy")
        history = (
            self._load_file("history.pkl")
            if model_type in ["cnn", "mlp", "xgboost"]
            else None
        )

        # Return all loaded components
        return (
            model,
            label_encoder,
            reference_data,
            scaler,
            is_scaler_fit,
            mix_included,
            highly_variable_genes,
            test_data,
            test_labels,
            history,
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
            exit()


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
        test_data,
        test_labels,
        label_encoder,
        reference_data,
        scaler,
        is_scaler_fit,
        highly_variable_genes,
        mix_included,
    ):
        """
        Initializes the ModelBuilder with the given configuration and training data.

        Parameters:
        - config (ConfigHandler): A ConfigHandler object containing configuration settings.
        - train_data (numpy.ndarray): The training data.
        - train_labels (numpy.ndarray): The labels for the training data.
        - test_data (numpy.ndarray): The test data.
        - test_labels (numpy.ndarray): The labels for the test data.
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
        self.test_data = test_data
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.reference_data = reference_data
        self.scaler = scaler
        self.is_scaler_fit = is_scaler_fit
        self.highly_variable_genes = highly_variable_genes
        self.mix_included = mix_included

    def create_cnn_model(self, num_output_units):
        """
        Create a Convolutional Neural Network (CNN) model using TensorFlow and the provided configuration.

        Args:
            num_output_units (int): The number of output units for the final layer of the model.

        Returns:
            model (tensorflow.python.keras.Model): The created and compiled CNN model.
        """

        cnn_config = self.config.ModelParameters.CNN_Model

        # Create model
        model = tf.keras.Sequential()

        # Convolutional blocks
        for i in range(len(cnn_config.filters)):
            model.add(
                tf.keras.layers.Conv1D(
                    filters=cnn_config.filters[i],
                    kernel_size=cnn_config.kernel_sizes[i],
                    strides=cnn_config.strides[i],
                    padding=cnn_config.paddings[i],
                    input_shape=(1, self.train_data.shape[2]),
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
                tf.keras.layers.Dense(
                    units=units, activation=cnn_config.activation_function
                )
            )
            model.add(tf.keras.layers.Dropout(rate=cnn_config.dropout_rate))
        model.add(tf.keras.layers.Dense(units=num_output_units, activation="softmax"))

        # Determine optimizer based on computer type
        if self.config.Device.processor.lower() == "m":
            optimizer_instance = tf.keras.optimizers.legacy.Adam(
                learning_rate=cnn_config.learning_rate
            )  # for M1/M2/M3 Macs
        else:
            optimizer_instance = tf.keras.optimizers.Adam(
                learning_rate=cnn_config.learning_rate
            )

        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss=self.config.Training.custom_loss,
            metrics=self.config.Training.metrics,
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
        mlp_config = self.config.ModelParameters.MLP_Model

        model = tf.keras.Sequential()

        # Input layer
        model.add(tf.keras.layers.InputLayer(input_shape=self.train_data.shape[1]))

        # Fully connected layers
        for units in mlp_config.units:
            model.add(
                tf.keras.layers.Dense(
                    units=units, activation=mlp_config.activation_function
                )
            )
            model.add(tf.keras.layers.Dropout(rate=mlp_config.dropout_rate))

        # Output layer
        model.add(tf.keras.layers.Dense(units=num_output_units, activation="softmax"))

        # Determine optimizer based on computer type
        if self.config.Device.processor.lower() == "m":
            optimizer_instance = tf.keras.optimizers.legacy.Adam(
                learning_rate=mlp_config.learning_rate
            )  # for M1/M2/M3 Macs
        else:
            optimizer_instance = tf.keras.optimizers.Adam(
                learning_rate=mlp_config.learning_rate
            )

        # Compile model
        model.compile(
            optimizer=optimizer_instance,
            loss=self.config.Training.custom_loss,
            metrics=self.config.Training.metrics,
        )

        return model

    def create_logistic_regression(self):
        """
        Create a logistic regression model using scikit-learn and the provided configuration.

        Returns:
            lr (sklearn.linear_model.LogisticRegression): The logistic regression model.
        """
        lr_config = self.config.ModelParameters.LogisticRegression_Model

        lr = LogisticRegression(
            penalty=lr_config.penalty,
            solver=lr_config.solver,
            l1_ratio=lr_config.l1_ratio if lr_config.penalty == "elasticnet" else None,
            random_state=lr_config.random_state,
            max_iter=lr_config.max_iter,
        )

        return lr

    def create_random_forest(self):
        """
        Create a random forest classifier using scikit-learn and the provided configuration.

        Returns:
            rf (sklearn.ensemble.RandomForestClassifier): The random forest classifier.
        """
        rf_config = self.config.ModelParameters.RandomForest_Model

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
        xgb_config = self.config.ModelParameters.XGBoost_Model

        # Determine task type and set corresponding XGBoost parameters
        if (
            self.config.DataParameters.GeneralSettings.encoding_variable.lower()
            == "sex"
        ):
            task_type = "binary"
            xgb_objective = "binary:logistic"
            eval_metric = "auc"
        else:
            task_type = "multiclass"
            xgb_objective = "multi:softmax"
            eval_metric = "mlogloss"

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

        # Adjust parameters for multiclass classification
        if task_type == "multiclass":
            xgb_params["num_class"] = len(np.unique(self.train_labels))

        # Initialize XGBoost model
        model = xgb.XGBClassifier(**xgb_params)
        model.set_params(early_stopping_rounds=xgb_config.early_stopping_rounds)

        return model

    def build_model(self):
        """
        Build a model based on the specified type in the config.

        Returns:
            model: The created model, which could be CNN, MLP, or logistic regression.
        """
        model_type = self.config.DataParameters.GeneralSettings.model_type.lower()
        num_output_units = (
            self.train_labels.shape[1] if model_type in ["cnn", "mlp"] else None
        )

        if model_type == "cnn":
            model = self.create_cnn_model(num_output_units)
            print(model.summary())
        elif model_type == "mlp":
            model = self.create_mlp_model(num_output_units)
            print(model.summary())
        elif model_type == "logisticregression":
            model = self.create_logistic_regression()
        elif model_type == "randomforest":
            model = self.create_random_forest()
        elif model_type == "xgboost":
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

        model_type = self.config.DataParameters.GeneralSettings.model_type.lower()

        if model_type in ["cnn", "mlp"]:
            history = self._train_neural_network(model, paths)
        else:
            history = self._train_sklearn_model(model, paths)

        return history, model

    def _prepare_directories(self):
        """
        Prepare directories for saving models and related artifacts.

        Returns:
            model_dir (str): The directory where the model and artifacts will be saved.
        """
        self.path_manager = PathManager(self.config)

        model_dir = self.path_manager.construct_model_directory()
        os.makedirs(model_dir, exist_ok=True)
        return model_dir

    def _define_paths(self, model_dir):
        """
        Define paths for saving model checkpoints and related artifacts.

        Args:
            model_dir (str): The directory where the model and artifacts will be saved.

        Returns:
            dict: A dictionary containing paths for various artifacts.
        """
        paths = {
            "label_path": os.path.join(model_dir, "label_encoder.pkl"),
            "reference_path": os.path.join(model_dir, "reference_data.npy"),
            "scaler_path": os.path.join(model_dir, "scaler.pkl"),
            "is_scaler_fit_path": os.path.join(model_dir, "is_scaler_fit.pkl"),
            "num_features_path": os.path.join(model_dir, "num_features.pkl"),
            "highly_variable_genes_path": os.path.join(
                model_dir, "highly_variable_genes.pkl"
            ),
            "mix_included_path": os.path.join(model_dir, "mix_included.pkl"),
            "test_data_path": os.path.join(model_dir, "test_data.npy"),
            "test_labels_path": os.path.join(model_dir, "test_labels.npy"),
            "history_path": os.path.join(model_dir, "history.pkl"),
            "model_dir": model_dir,
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
        custom_model_path = os.path.join(paths["model_dir"], "best_model.h5")
        best_val_loss_path = os.path.join(paths["model_dir"], "best_val_loss.json")

        # Load the best validation loss from file
        try:
            with open(best_val_loss_path, "r") as f:
                best_val_loss = json.load(f)["best_val_loss"]
        except FileNotFoundError:
            print("No best validation loss found. Starting from scratch.")
            best_val_loss = float("inf")

        # Split data into training and validation sets
        (
            train_inputs_split,
            val_inputs_split,
            train_labels_split,
            val_labels_split,
        ) = train_test_split(
            self.train_data,
            self.train_labels,
            test_size=self.config.DataSplit.validation_split,
            random_state=self.config.DataSplit.random_state,
            stratify=self.train_labels,
        )

        # Define callbacks for early stopping and model saving
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=self.config.Training.early_stopping_patience
        )
        model_checkpoint = CustomModelCheckpoint(
            custom_model_path,
            best_val_loss_path,
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
            self.test_data,
            self.test_labels,
            paths["test_data_path"],
            paths["test_labels_path"],
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        )
        model_checkpoint.set_best_val_loss(best_val_loss)

        # Fit the model with validation split
        history = model.fit(
            train_inputs_split,
            train_labels_split,
            epochs=self.config.Training.epochs,
            batch_size=self.config.Training.batch_size,
            validation_data=(val_inputs_split, val_labels_split),
            callbacks=[early_stopping, model_checkpoint],
        )

        # Save the history object only if the model was improved and saved at least once during training
        if model_checkpoint.model_improved:
            print("saving history")
            with open(paths["history_path"], "wb") as f:
                pickle.dump(history.history, f)

        return history

    def _train_sklearn_model(self, model, paths):
        """
        Train a scikit-learn model (Logistic Regression, Random Forest, or XGBoost) and save the best model based on validation accuracy.

        Args:
            model: The scikit-learn model to train.
            paths (dict): A dictionary containing paths for saving artifacts.

        Returns:
            history: The training history (if applicable).
        """
        model_type = self.config.DataParameters.GeneralSettings.model_type.lower()
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
            test_size=self.config.DataSplit.validation_split,
            random_state=self.config.DataSplit.random_state,
            stratify=train_labels,
        )

        if model_type in ["logisticregression", "randomforest"]:
            model.fit(train_inputs_split, train_labels_split)
            history = None
        elif model_type == "xgboost":
            # Training with evaluation set
            eval_set = [
                (train_inputs_split, train_labels_split),
                (val_inputs_split, val_labels_split),
            ]
            model.fit(
                train_inputs_split,
                train_labels_split,
                eval_set=eval_set,
                verbose=True,
            )
            history = model.evals_result()
        else:
            raise ValueError("Unsupported model type provided.")

        # Evaluate the model
        val_predictions = model.predict(val_inputs_split)
        val_accuracy = accuracy_score(val_labels_split, val_predictions)

        # Load the best validation accuracy from file, if it exists
        best_val_accuracy_path = os.path.join(
            paths["model_dir"], "best_val_accuracy.json"
        )

        try:
            with open(best_val_accuracy_path, "r") as f:
                best_val_accuracy = json.load(f)["best_val_accuracy"]
        except FileNotFoundError:
            best_val_accuracy = 0  # Initialize if no record exists

        # Check if the current model outperforms the best one so far
        print("Validation accuracy:", val_accuracy)
        print("Best validation accuracy so far:", best_val_accuracy)
        print("Model improved:", val_accuracy > best_val_accuracy)

        if val_accuracy > best_val_accuracy:
            # Update the record with the new best validation accuracy
            with open(best_val_accuracy_path, "w") as f:
                json.dump({"best_val_accuracy": val_accuracy}, f)

            # Save selected features for reference in the future
            num_features = self.train_data.shape[1]
            with open(paths["num_features_path"], "wb") as f:
                pickle.dump(num_features, f)

            # Save the model as it's the best one so far using pickle
            model_path = os.path.join(paths["model_dir"], "best_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            # Save the history object
            if model_type == "xgboost" and history is not None:
                with open(paths["history_path"], "wb") as f:
                    pickle.dump(history, f)

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

            # Save test data and labels
            np.save(paths["test_data_path"], self.test_data)
            np.save(paths["test_labels_path"], self.test_labels)

            print("New best model saved with validation accuracy:", val_accuracy)

        return history

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
        history, model = self.train_model(model)

        return model, history
