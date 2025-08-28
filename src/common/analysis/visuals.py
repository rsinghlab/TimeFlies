import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy.sparse import issparse
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from common.utils.logging_config import get_logger

logger = get_logger(__name__)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


class VisualizationTools:
    """
    This class provides methods for visualizing various aspects of machine learning model results.

    Args:
        config (ConfigHandler): Configuration handler object with nested configuration attributes.
        path_manager (PathManager): Path manager object to handle directory paths.

    Methods:
        plot_class_distribution(class_counts, file_name, dataset, subfolder_name):
            Plots the distribution of the classes and saves the plot.

        create_styled_dataframe(df, subfolder_name, dataset, file_name):
            Create a styled dataframe for gene expression data and save it as an image.

        visualize_sparse_vs_dense(adata, subset_size, head_size, file_name, dataset, subfolder_name):
            Visualizes and compares sparse vs dense representations of data.

        create_confusion_matrix(y_true, y_pred, class_names, file_name, subfolder_name):
            Creates and saves a confusion matrix plot based on the true and predicted labels.

        plot_history(history, file_name, subfolder_name):
            Plots the training and validation loss, accuracy, and AUC for each epoch in a given Keras training history.

        plot_xgboost_history(history, file_name, subfolder_name):
            Plots the training and validation metrics for each epoch in an XGBoost training history.

        plot_roc_curve(y_true, y_score, n_classes, class_names, file_name, subfolder_name):
            Creates and saves a ROC curve plot based on the true labels and score predicted by the model.

        plot_shap_summary(shap_values, test_data, feature_names, class_names, file_name_prefix):
            Plots the SHAP summary plot for the given SHAP values and test data.
    """

    def __init__(self, config, path_manager, output_dir=None):
        """
        Initializes the VisualizationTools with the given configuration, path manager, and model version.

        Parameters:
        - config (ConfigHandler): Configuration handler object with nested configuration attributes.
        - path_manager (PathManager): The path manager object for directory paths.
        - output_dir (str, optional): Explicit output directory. If None, uses path_manager default.
        """
        self.config = config
        self.path_manager = path_manager

        # Use explicit output directory or fall back to path manager
        if output_dir:
            self.output_dir = output_dir
        else:
            # For backward compatibility, use old structure
            self.output_dir = self.path_manager.get_visualization_directory()

    def plot_class_distribution(self, class_counts, file_name, dataset, subfolder_name):
        """
        Plots the distribution of the classes and saves the plot.

        Args:
            class_counts (pandas.Series): Series with class counts.
            file_name (str): The name of the file to save the plot.
            dataset (str): Name of the dataset.
            subfolder_name (str): Name of the subfolder to save the plot.

        Returns:
            None
        """
        # Plotting the class distribution
        plt.figure(figsize=(10, 5))
        ax = sns.barplot(
            x=class_counts.index,
            y=class_counts.values,
            alpha=0.8,
            order=class_counts.index,
        )

        # Add count annotations to the bars
        for p in ax.patches:
            ax.annotate(
                f"{int(p.get_height())}",
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        # Set the title, x- and y-axis labels, and rotate the x-axis labels for better visibility
        plt.title("Class Distribution")
        plt.ylabel("Number of Occurrences", fontsize=12)
        plt.xlabel("Class", fontsize=12)
        plt.xticks(rotation=45)

        # Create a new subfolder with the specified name
        subfolder_dir = os.path.join(self.output_dir, "..", subfolder_name, dataset)
        os.makedirs(subfolder_dir, exist_ok=True)

        # Save the plot
        output_file_path = os.path.join(subfolder_dir, file_name)
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close()

    def create_styled_dataframe(self, df, subfolder_name, dataset, file_name):
        """
        Create a styled dataframe for gene expression data and save it as an image using Matplotlib.

        Args:
            df (pandas.DataFrame): DataFrame containing gene expression data.
            subfolder_name (str): Name of the subfolder to save the styled DataFrame image.
            dataset (str): Name of the dataset.
            file_name (str): The name of the file to save the styled DataFrame image.

        Returns:
            None
        """
        # Select the first 5 rows (or as needed)
        df_to_export = df.head()

        # Create a new subfolder with the specified name
        subfolder_dir = os.path.join(self.output_dir, "..", subfolder_name, dataset)
        os.makedirs(subfolder_dir, exist_ok=True)

        # Create a Matplotlib figure and axis
        fig, ax = plt.subplots(
            figsize=(len(df_to_export.columns) * 1.2, len(df_to_export) * 0.6 + 1)
        )  # Adjust size as needed
        ax.axis("tight")
        ax.axis("off")

        # Create the table
        table = ax.table(
            cellText=df_to_export.values,
            colLabels=df_to_export.columns,
            cellLoc="center",
            loc="center",
        )

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width(col=list(range(len(df_to_export.columns))))

        # Set header background color
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#606060")
                cell.set_text_props(color="white", fontfamily="sans-serif")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(color="black", fontfamily="sans-serif")

        # Adjust layout
        plt.tight_layout()

        # Save the table as an image
        output_file_path = os.path.join(subfolder_dir, f"{file_name}.png")
        plt.savefig(output_file_path, bbox_inches="tight", dpi=300)
        plt.close()

    def visualize_sparse_vs_dense(
        self, adata, subset_size, head_size, file_name, dataset, subfolder_name
    ):
        """
        Visualize and compare sparse vs dense representations of data.

        Args:
            adata (AnnData): AnnData object containing the dataset.
            subset_size (int): Number of samples to subset.
            head_size (int): Number of samples to display.
            file_name (str): File name to save the plot.
            dataset (str): Name of the dataset.
            subfolder_name (str): Name of the subfolder to save the plot.

        Returns:
            None
        """
        # Extract a subset of the data
        subset_data = adata.X[:subset_size, :subset_size]

        print("\nSparse Matrix Head (Non-zero values):")
        print(pd.DataFrame(subset_data).head(head_size))

        # Convert to dense if it's sparse
        if issparse(subset_data):
            dense_data = subset_data.toarray()
        else:
            dense_data = subset_data

        # Print the head of both dense and sparse matrices
        print("Dense Matrix Head:")
        print(pd.DataFrame(dense_data).head(head_size))

        # Plotting the dense data
        plt.figure(figsize=(20, 10))
        plt.subplot(1, 2, 1)
        sns.heatmap(
            dense_data[:head_size, :head_size], cmap="viridis", cbar=False, annot=True
        )
        plt.title("Dense Representation")

        # Plotting the sparse data (using non-zero structure)
        plt.subplot(1, 2, 2)
        plt.spy(subset_data[:head_size, :head_size], markersize=10)
        plt.title("Sparse Representation (Non-zero structure)")

        # Comparing memory usage
        dense_memory = sys.getsizeof(dense_data)
        sparse_memory = (
            sys.getsizeof(subset_data.data)
            if issparse(subset_data)
            else sys.getsizeof(subset_data)
        )
        print(f"Memory used by dense matrix: {dense_memory / 1e6:.2f} MB")
        print(f"Memory used by sparse matrix: {sparse_memory / 1e6:.2f} MB")

        # Create a new subfolder with the specified name
        subfolder_dir = os.path.join(self.output_dir, "..", subfolder_name, dataset)
        os.makedirs(subfolder_dir, exist_ok=True)

        # Save the plot
        output_file_path = os.path.join(subfolder_dir, file_name)
        plt.tight_layout()
        plt.savefig(output_file_path)
        plt.close()

    def create_confusion_matrix(self, y_true, y_pred, class_names, file_name):
        """
        Creates and saves a confusion matrix plot based on the true and predicted labels.

        Args:
            y_true (array-like): Ground truth (correct) target values.
            y_pred (array-like): Estimated targets as returned by a classifier.
            class_names (list): List of class names (strings) in order of the confusion matrix.
            file_name (str): File name to save the confusion matrix plot.

        Returns:
            None
        """
        # Compute the confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Create a ConfusionMatrixDisplay object using the computed confusion matrix and class names
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

        # Create a new Matplotlib figure with a given size
        fig, ax = plt.subplots(figsize=(8, 8))

        # Use the ConfusionMatrixDisplay object to plot the confusion matrix
        disp.plot(ax=ax, cmap="Blues", xticks_rotation="vertical")

        # Set the x- and y-axis labels and title of the plot
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")

        # Save the plot to a file in the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        output_file_path = os.path.join(self.output_dir, file_name)
        plt.tight_layout()
        fig.savefig(output_file_path)

        # Close the figure to free up memory
        plt.close(fig)

    def plot_history(self, history, file_name):
        """
        Plots the training and validation loss, accuracy, AUC (Area Under the ROC Curve),
        for each epoch in a given Keras training history.

        Args:
            history (tensorflow.python.keras.callbacks.History or dict): A Keras training history object or a
            dictionary containing training history data.
            file_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Check if the input is a Keras training history object or a dictionary containing training history data
        if isinstance(history, dict):
            history_data = history
        else:
            history_data = history.history

        # Define the metrics to plot and their titles and y-axis labels
        # Try different variations of AUC metric names
        auc_metric = None
        for auc_name in ["auc", "AUC", "val_auc", "val_AUC"]:
            if auc_name in history_data:
                auc_metric = auc_name.replace("val_", "")  # Use the base metric name
                break

        metrics = [
            ("loss", "Model Loss", "Loss"),
            ("accuracy", "Model Accuracy", "Accuracy"),
        ]

        if auc_metric:
            metrics.append((auc_metric, "Model AUC", "AUC"))

        # Determine the number of rows and columns for the subplot grid
        num_metrics = len(metrics)
        ncols = 3
        nrows = (num_metrics + ncols - 1) // ncols

        # Create a new Matplotlib figure with the appropriate number of subplots
        fig, axes = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows)
        )
        axes = axes.flatten()

        # For each metric, plot the train and validation data, and set the title and y-axis label
        plot_index = 0
        for i, (metric, title, ylabel) in enumerate(metrics):
            # Skip metrics that don't exist in the history
            if metric not in history_data:
                continue

            ax = axes[plot_index]
            ax.plot(range(1, len(history_data[metric]) + 1), history_data[metric])
            ax.plot(
                range(1, len(history_data.get(f"val_{metric}", [])) + 1),
                history_data.get(f"val_{metric}", []),
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Epoch")
            ax.legend(["Train", "Validation"], loc="best")
            plot_index += 1

        # Remove any extra subplots
        for i in range(plot_index, len(axes)):
            fig.delaxes(axes[i])

        # Create a new subfolder with the specified name
        subfolder_dir = os.path.join(self.output_dir, self.output_dir)
        os.makedirs(subfolder_dir, exist_ok=True)

        # Save the figure to a file in the output directory and close the figure to free up memory
        output_file_path = os.path.join(subfolder_dir, file_name)
        plt.savefig(output_file_path)
        plt.close()

    def plot_xgboost_history(self, history, file_name):
        """
        Plots the training and validation metrics for each epoch in an XGBoost training history.

        Args:
            history (dict): An XGBoost training history object.
            file_name (str): The name of the file to save the plot.

        Returns:
            None
        """
        # Modify the label names for clarity in plots
        label_modifications = {"validation_0": "Train", "validation_1": "Validation"}

        # Plot each metric in the history
        for metric in history["validation_0"].keys():
            epochs = len(history["validation_0"][metric])
            x_axis = range(1, epochs + 1)

            plt.figure(figsize=(10, 6))
            for val in history.keys():
                plt.plot(
                    x_axis,
                    history[val][metric],
                    label=label_modifications.get(val, val),
                )
            plt.title(f"Model {metric.upper()}")
            plt.ylabel(metric.upper())
            plt.xlabel("Epoch")
            plt.legend()

            # Save the plot to a file in the output directory and close the figure to free up memory
            os.makedirs(self.output_dir, exist_ok=True)
            output_file_path = os.path.join(self.output_dir, file_name)
            plt.savefig(output_file_path)
            plt.close()

    def plot_roc_curve(self, y_true, y_score, n_classes, class_names, file_name):
        """
        Creates and saves a ROC curve plot based on the true labels and score predicted by the model.

        Args:
            y_true (array-like): Ground truth (correct) target values.
            y_score (array-like): Target scores.
            n_classes (int): Number of classes.
            class_names (list): List of class names (strings) in order of the confusion matrix.
            file_name (str): File name to save the ROC curve plot.

        Returns:
            None
        """
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            if n_classes == 2:  # Binary classification
                # Handle case where predictions might have 1 or 2 columns
                if y_score.shape[1] > 1:
                    y_score_positive = y_score[:, 1]  # Use positive class column
                else:
                    y_score_positive = y_score[:, 0]  # Use single column

                # For binary, y_true might be 1D or 2D
                y_true_flat = y_true.flatten() if len(y_true.shape) > 1 else y_true
                fpr[i], tpr[i], _ = roc_curve(y_true_flat, y_score_positive)
            else:  # Multi-class classification
                # Handle cases where arrays might have fewer columns than expected
                if i < y_true.shape[1] and i < y_score.shape[1]:
                    y_true_class = y_true[:, i]
                    y_score_class = y_score[:, i]
                    if np.sum(y_true_class) == 0:
                        # Avoid errors if a class is not present in y_true
                        fpr[i], tpr[i], _ = (np.array([0, 1]), np.array([0, 1]), None)
                    else:
                        fpr[i], tpr[i], _ = roc_curve(y_true_class, y_score_class)
                else:
                    # Skip classes that don't exist in the data
                    fpr[i], tpr[i], _ = (np.array([0, 1]), np.array([0, 1]), None)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute macro-average ROC curve and ROC area and Aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Interpolate all ROC curves
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        # Average and compute AUC
        mean_tpr /= n_classes
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        # Plot ROC curve
        plt.figure()
        plt.plot(
            fpr["macro"],
            tpr["macro"],
            label="macro-average ROC curve (area = {:0.2f})".format(roc_auc["macro"]),
        )
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label=f"ROC curve of {class_names[i]} (area = {roc_auc[i]:0.2f})",
            )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve: Multi-class")
        plt.legend(loc="lower right")

        # Save the plot
        os.makedirs(self.output_dir, exist_ok=True)
        output_file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(output_file_path)
        plt.close()

    def plot_shap_summary(
        self,
        shap_values,
        test_data,
        feature_names,
        class_names,
        file_name_prefix,
    ):
        """
        Plots SHAP summary plots for the given SHAP values and test data.

        Args:
            shap_values (array-like): SHAP values computed for the test data.
            test_data (array-like): Test data for which SHAP values were computed.
            feature_names (array-like): Names of the features corresponding to the columns of test_data.
            class_names (array-like): Names of the classes.
            file_name_prefix (str): Prefix for the saved files. Individual class plots will append the class name.

        Returns:
            None
        """
        # Path for saving plots
        output_subfolder = self.path_manager.get_visualization_directory(
            subfolder="SHAP"
        )
        os.makedirs(output_subfolder, exist_ok=True)

        # Select the relevant feature names
        relevant_feature_names = feature_names[: test_data.shape[1]]

        # If multi-class, save individual class SHAP plots
        if isinstance(shap_values, list):
            for index, class_name in enumerate(class_names):
                shap.summary_plot(
                    shap_values[index],
                    test_data,
                    feature_names=feature_names,
                    show=False,
                )
                plt.title(f"SHAP Summary Plot - Class: {class_name}", fontsize=16)
                plt.ylabel("Genes", fontsize=12)
                plt.tight_layout()
                plt.savefig(
                    os.path.join(
                        output_subfolder, f"{file_name_prefix}_{class_name}.png"
                    )
                )
                plt.close()

        # Generate and save the overall SHAP summary plot (multi-class or binary)
        shap.summary_plot(
            shap_values,
            test_data,
            feature_names=relevant_feature_names,
            class_names=class_names,
            show=False,
        )
        plt.title("SHAP Summary Plot", fontsize=16)
        plt.ylabel("Genes", fontsize=12)
        plt.xlabel("mean(|SHAP Value|)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_subfolder, f"{file_name_prefix}_Overall.png"))
        plt.close()


class Visualizer:
    """
    A class to handle visualization of model results and explanations.

    This class generates visualizations for the model's performance and the explanations
    generated using SHAP.
    """

    def __init__(
        self,
        config,
        model,
        history,
        test_inputs,
        test_labels,
        label_encoder,
        squeezed_shap_values,
        squeezed_test_data,
        adata,
        adata_corrected,
        path_manager,
        preserved_var_names=None,
    ):
        """
        Initializes the Visualizer with the given configuration and results.

        Parameters:
        - config (ConfigHandler): A configuration handler object with nested configuration attributes.
        - model (object): The trained model.
        - history (History): The training history object.
        - test_inputs (numpy.ndarray): The test data.
        - test_labels (numpy.ndarray): The labels for the test data.
        - label_encoder (LabelEncoder): The label encoder used during training.
        - squeezed_shap_values (numpy.ndarray): SHAP values for the test data.
        - squeezed_test_data (numpy.ndarray): Test data corresponding to the SHAP values.
        - adata (AnnData): The main AnnData object containing the dataset.
        - adata_corrected (AnnData): The batch-corrected AnnData object.
        """
        self.config = config
        self.model = model
        self.history = history
        self.test_inputs = test_inputs
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.squeezed_shap_values = squeezed_shap_values
        self.squeezed_test_data = squeezed_test_data
        self.adata = adata
        self.adata_corrected = adata_corrected
        self.preserved_var_names = preserved_var_names

        # Initialize PathManager
        self.path_manager = path_manager

        # Initialize VisualizationTools for evaluation visuals (will be updated based on context)
        # Don't create visual_tools yet - it will be created when context is set
        self.visual_tools = None

        # Training visual tools will be created only when needed
        self.training_visual_tools = None

    def set_evaluation_context(self, experiment_name=None):
        """
        Set the context for evaluation visuals using experiment structure.

        Args:
            experiment_name: Specific experiment name to use
        """
        # Store experiment name for use in other methods
        self.experiment_name = experiment_name

        if experiment_name:
            experiment_dir = self.path_manager.get_experiment_dir(experiment_name)
        else:
            # For evaluation, use the best trained experiment instead of creating new one
            try:
                best_experiment = self.path_manager.get_best_experiment_name()
                experiment_dir = self.path_manager.get_experiment_dir(best_experiment)
            except (FileNotFoundError, RuntimeError):
                # Fallback: create new experiment if no trained models exist
                experiment_dir = self.path_manager.get_experiment_dir()

        eval_visuals_dir = os.path.join(experiment_dir, "evaluation", "plots")
        os.makedirs(eval_visuals_dir, exist_ok=True)
        self.visual_tools = VisualizationTools(
            self.config, self.path_manager, eval_visuals_dir
        )

    def import_metrics(self):
        """
        Sets y_true, y_pred, y_pred_class (for classification) or y_true, y_pred (for regression)
        """
        model_type = getattr(self.config.data, "model", "CNN").lower()
        task_type = getattr(self.config.model, "task_type", "classification")

        if model_type in ["mlp", "cnn"]:
            self.y_pred = self.model.predict(self.test_inputs, verbose=0)
        else:
            if task_type == "regression":
                self.y_pred = self.model.predict(self.test_inputs)
            else:
                self.y_pred = self.model.predict_proba(self.test_inputs)

        if task_type == "regression":
            # For regression, no class conversion needed
            self.y_true = self.test_labels.flatten()
            self.y_pred = self.y_pred.flatten() if len(self.y_pred.shape) > 1 else self.y_pred
        else:
            # For classification, convert predictions and true labels to class indices
            self.y_pred_class = np.argmax(self.y_pred, axis=1)
            self.y_true_class = np.argmax(self.test_labels, axis=1)

    def _visualize_training_history(self):
        """
        Visualize the training history for the model based on the configuration.
        Saves to models/.../training/visuals/ directory.
        """
        # Create training visual tools only when actually needed
        if self.training_visual_tools is None:
            # Use stored experiment name if available
            experiment_name = getattr(self, "experiment_name", None)
            training_visuals_dir = self.path_manager.get_training_visuals_dir(
                experiment_name
            )
            self.training_visual_tools = VisualizationTools(
                self.config, self.path_manager, training_visuals_dir
            )

        model_type = getattr(self.config.data, "model", "CNN").lower()
        if model_type in ["mlp", "cnn"]:
            self.training_visual_tools.plot_history(
                self.history, "training_metrics.png"
            )
        elif model_type == "xgboost":
            self.training_visual_tools.plot_xgboost_history(
                self.history, "training_metrics.png"
            )

    def _visualize_confusion_matrix(self):
        """
        Visualize the confusion matrix for the predicted and true labels.
        """
        # Get class labels from label encoder
        class_labels = self.label_encoder.classes_

        # Sort the class labels based on age if specified in the config
        if "age" in getattr(self.config.data, "target_variable", "age"):
            class_labels = self._sort_labels_by_age(class_labels)

        # Create confusion matrix
        self.visual_tools.create_confusion_matrix(
            self.y_true_class,
            self.y_pred_class,
            class_labels,
            "confusion_matrix.png",
        )

    def _sort_labels_by_age(self, class_labels):
        """
        Sort class labels by age if encoding variable contains age.
        """
        ages = []
        for label in class_labels:
            # First try to parse the whole label as an integer
            if str(label).isdigit():
                ages.append(int(label))
            else:
                # If that fails, try splitting by underscore and get the numeric part
                split_label = str(label).split("_")
                if len(split_label) > 1 and split_label[1].isdigit():
                    ages.append(int(split_label[1]))
                else:
                    # Fall back to -1 for non-numeric labels
                    ages.append(-1)
        sorted_labels = [
            pair[0] for pair in sorted(zip(class_labels, ages), key=lambda x: x[1])
        ]
        return sorted_labels

    def _plot_roc_curve(self):
        """
        Plot ROC curve based on the predictions and true labels.
        """
        y_true_binary = label_binarize(
            self.y_true_class, classes=np.unique(self.y_true_class)
        )
        y_pred_prob = self.y_pred

        n_classes = len(self.label_encoder.classes_)

        self.visual_tools.plot_roc_curve(
            y_true_binary,
            y_pred_prob,
            n_classes,
            self.label_encoder.classes_,
            "roc_curve.png",
        )

    def _plot_shap_summary(self):
        """
        Generate SHAP summary plot if SHAP values are available.
        """
        if self.squeezed_shap_values is not None:
            # Try to get var_names from adata objects first
            var_names = None
            if getattr(self.config.data.batch_correction, "enabled", False) or getattr(
                self.config.preprocessing.genes,
                "select_batch_genes",
                False,
            ):
                if self.adata_corrected is not None:
                    var_names = self.adata_corrected.var_names
            else:
                if self.adata is not None:
                    var_names = self.adata.var_names

            # If adata objects are not available, use preserved gene names (evaluation mode)
            if var_names is None and self.preserved_var_names is not None:
                var_names = self.preserved_var_names

            # Only proceed if we have actual gene names
            if var_names is None:
                logger.warning(
                    "No gene names available for SHAP visualization - skipping SHAP plots"
                )
                return
            # Get class names only for classification tasks
            class_names = None
            if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                class_names = self.label_encoder.classes_

            self.visual_tools.plot_shap_summary(
                shap_values=self.squeezed_shap_values,
                test_data=self.squeezed_test_data,
                feature_names=var_names,
                class_names=class_names,
                file_name_prefix="SHAP_Summary",
            )

    def _plot_regression_metrics(self):
        """
        Generate regression-specific plots: predicted vs actual, residuals, and distribution plots.
        """
        # Create a 2x2 subplot for comprehensive regression analysis
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Regression Model Evaluation', fontsize=16, fontweight='bold')

        # 1. Predicted vs Actual scatter plot
        ax1 = axes[0, 0]
        ax1.scatter(self.y_true, self.y_pred, alpha=0.6, color='blue', s=30)

        # Perfect prediction line (y = x)
        min_val = min(min(self.y_true), min(self.y_pred))
        max_val = max(max(self.y_true), max(self.y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Add correlation coefficient
        correlation = np.corrcoef(self.y_true, self.y_pred)[0, 1]
        ax1.text(0.05, 0.95, f'R = {correlation:.3f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # 2. Residuals vs Predicted plot
        ax2 = axes[0, 1]
        residuals = self.y_pred - self.y_true
        ax2.scatter(self.y_pred, residuals, alpha=0.6, color='green', s=30)
        ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals (Predicted - Actual)')
        ax2.set_title('Residuals vs Predicted')
        ax2.grid(True, alpha=0.3)

        # Add standard deviation lines
        residual_std = np.std(residuals)
        ax2.axhline(y=residual_std, color='orange', linestyle=':', alpha=0.7, label='±1 STD')
        ax2.axhline(y=-residual_std, color='orange', linestyle=':', alpha=0.7)
        ax2.axhline(y=2*residual_std, color='red', linestyle=':', alpha=0.7, label='±2 STD')
        ax2.axhline(y=-2*residual_std, color='red', linestyle=':', alpha=0.7)
        ax2.legend()

        # 3. Residuals histogram
        ax3 = axes[1, 0]
        ax3.hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)

        # Add normal distribution overlay
        mu, sigma = np.mean(residuals), np.std(residuals)
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_dist = ((1/(sigma * np.sqrt(2 * np.pi))) *
                      np.exp(-0.5 * ((x - mu) / sigma) ** 2))
        # Scale to match histogram
        normal_dist *= len(residuals) * (residuals.max() - residuals.min()) / 30
        ax3.plot(x, normal_dist, 'r-', linewidth=2, label='Normal Distribution')
        ax3.legend()

        # 4. Q-Q plot for residuals normality
        ax4 = axes[1, 1]
        from scipy import stats as scipy_stats
        scipy_stats.probplot(residuals, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Residuals Normality)')
        ax4.grid(True, alpha=0.3)

        # Add metrics text box
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(self.y_true, self.y_pred)
        rmse = np.sqrt(mean_squared_error(self.y_true, self.y_pred))
        r2 = r2_score(self.y_true, self.y_pred)

        metrics_text = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
        fig.text(0.02, 0.02, metrics_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.subplots_adjust(top=0.93)  # Make room for main title

        # Save the plot
        output_file_path = os.path.join(self.visual_tools.output_dir, "regression_evaluation.png")
        plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def run(self):
        """
        Run the full visualization pipeline (evaluation-only, no training plots).
        """
        self.import_metrics()

        # Get task type from config
        task_type = getattr(self.config.model, "task_type", "classification")

        if task_type == "classification":
            # Classification-specific visualizations
            self._visualize_confusion_matrix()
            self._plot_roc_curve()
        else:
            # Regression-specific visualizations
            self._plot_regression_metrics()

        # SHAP summary works for both classification and regression
        self._plot_shap_summary()

    def run_with_training(self):
        """
        Run visualization pipeline including training plots.
        """
        self._visualize_training_history()
        self.run()  # Run evaluation visualizations
