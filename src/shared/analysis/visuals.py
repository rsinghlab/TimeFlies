import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
)
import seaborn as sns
from sklearn.preprocessing import label_binarize
import shap
import sys
import pandas as pd
from scipy.sparse import issparse
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning

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

    def __init__(self, config, path_manager):
        """
        Initializes the VisualizationTools with the given configuration, path manager, and model version.

        Parameters:
        - config (ConfigHandler): Configuration handler object with nested configuration attributes.
        - path_manager (PathManager): The path manager object for directory paths.
        """
        self.config = config
        self.path_manager = path_manager

        # Determine the main output directory using PathManager
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
                y_score_positive = y_score[:, 1]
                fpr[i], tpr[i], _ = roc_curve(y_true, y_score_positive)
            else:  # Multi-class classification
                y_true_class = y_true[:, i]
                y_score_class = y_score[:, i]
                if np.sum(y_true_class) == 0:
                    # Avoid errors if a class is not present in y_true
                    fpr[i], tpr[i], _ = (np.array([0, 1]), np.array([0, 1]), None)
                else:
                    fpr[i], tpr[i], _ = roc_curve(y_true_class, y_score_class)
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
            label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
        )
        for i in range(n_classes):
            plt.plot(
                fpr[i],
                tpr[i],
                label="ROC curve of {0} (area = {1:0.2f})".format(
                    class_names[i], roc_auc[i]
                ),
            )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve: Multi-class")
        plt.legend(loc="lower right")

        # Save the plot
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

        # Initialize VisualizationTools with PathManager
        self.visual_tools = VisualizationTools(self.config, self.path_manager)

    def import_metrics(self):
        """
        Sets y_true, y_pred, y_pred_class
        """
        model_type = getattr(self.config.data, "model", "CNN").lower()
        if model_type in ["mlp", "cnn"]:
            self.y_pred = self.model.predict(self.test_inputs)
        else:
            self.y_pred = self.model.predict_proba(self.test_inputs)

        # Convert predictions and true labels to class indices
        self.y_pred_class = np.argmax(self.y_pred, axis=1)
        self.y_true_class = np.argmax(self.test_labels, axis=1)

    def _visualize_training_history(self):
        """
        Visualize the training history for the model based on the configuration.
        """
        model_type = getattr(self.config.data, "model", "CNN").lower()
        if model_type in ["mlp", "cnn"]:
            self.visual_tools.plot_history(self.history, "training_metrics.png")
        elif model_type == "xgboost":
            self.visual_tools.plot_xgboost_history(self.history, "training_metrics.png")

    def _visualize_confusion_matrix(self):
        """
        Visualize the confusion matrix for the predicted and true labels.
        """
        # Get class labels from label encoder
        class_labels = self.label_encoder.classes_

        # Sort the class labels based on age if specified in the config
        if "age" in getattr(self.config.data, "encoding_variable", "age"):
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
            self.visual_tools.plot_shap_summary(
                shap_values=self.squeezed_shap_values,
                test_data=self.squeezed_test_data,
                feature_names=var_names,
                class_names=self.label_encoder.classes_,
                file_name_prefix="SHAP_Summary",
            )

    def run(self):
        """
        Run the visualization pipeline.
        """
        self._visualize_training_history()
        self.import_metrics()
        self._visualize_confusion_matrix()
        self._plot_roc_curve()
        self._plot_shap_summary()
