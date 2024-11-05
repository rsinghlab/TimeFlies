import logging

from config import config
from preprocess import DataPreprocessor, GeneFilter
from interpreter import Interpreter, Metrics
from model import ModelBuilder, ModelLoader
from visuals import Visualizer
from eda import EDAHandler
from utilities import GPUHandler, PathManager, DataLoader

logger = logging.getLogger(__name__)


class PipelineManager:
    """
    The PipelineManager class orchestrates the entire data processing, modeling,
    and visualization pipeline. It handles tasks such as GPU configuration, data loading,
    preprocessing, model training/loading, interpretation, and visualization based on the
    configurations provided.

    Attributes:
        config_instance (ConfigHandler): The configuration object containing all settings.
        data_loader (DataLoader): Instance responsible for loading data.
        path_manager (PathManager): Manages file paths based on the configuration.
    """

    def __init__(self):
        """
        Initialize the PipelineManager class with configuration and data loader.
        """
        self.config_instance = config
        self.data_loader = DataLoader(self.config_instance)
        self.path_manager = PathManager(self.config_instance)
        logger.info("Initializing TimeFlies pipeline...")

    def setup_gpu(self):
        """
        Configure TensorFlow GPU settings if available.
        """
        try:
            logger.info("Setting up GPU...")
            GPUHandler.configure(self.config_instance)
            logger.info("GPU configuration successful.")
        except Exception as e:
            logger.error(f"Error configuring GPU: {e}")
            raise

    def load_data(self):
        """
        Load data and gene lists.
        """
        try:
            logger.info("Loading data...")
            (
                self.adata,
                self.adata_eval,
                self.adata_original,
            ) = self.data_loader.load_data()
            (
                self.adata_corrected,
                self.adata_eval_corrected,
            ) = self.data_loader.load_corrected_data()
            self.autosomal_genes, self.sex_genes = self.data_loader.load_gene_lists()
            logger.info("Data loading complete.")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def run_eda(self):
        """
        Perform Exploratory Data Analysis (EDA) if specified in the config.
        """
        try:
            if self.config_instance.DataProcessing.ExploratoryDataAnalysis.enabled:
                logger.info("Running Exploratory Data Analysis (EDA)...")
                eda_handler = EDAHandler(
                    self.config_instance,
                    self.path_manager,
                    self.adata,
                    self.adata_eval,
                    self.adata_original,
                    self.adata_corrected,
                    self.adata_eval_corrected,
                )
                eda_handler.run_eda()
                logger.info("EDA completed.")
            else:
                logger.info("Skipping EDA as it is disabled in the configuration.")
        except Exception as e:
            logger.error(f"Error during EDA: {e}")
            raise

    def setup_gene_filtering(self):
        """
        Filter genes using the GeneFilter class.
        """
        try:
            logger.info("Applying gene filtering...")
            self.gene_filter = GeneFilter(
                self.config_instance,
                self.adata,
                self.adata_eval,
                self.adata_original,
                self.autosomal_genes,
                self.sex_genes,
            )
            (
                self.adata,
                self.adata_eval,
                self.adata_original,
            ) = self.gene_filter.apply_filter()
            logger.info("Gene filtering complete.")
        except Exception as e:
            logger.error(f"Error during gene filtering: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocess general training and test data for building or training a new model.
        """
        try:
            logger.info("Preprocessing data for training and testing...")
            self.data_preprocessor = DataPreprocessor(
                self.config_instance, self.adata, self.adata_corrected
            )
            (
                self.train_data,
                self.test_data,
                self.train_labels,
                self.test_labels,
                self.label_encoder,
                self.reference_data,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.mix_included,
            ) = self.data_preprocessor.prepare_data()
            logger.info("Training and testing data preprocessed successfully.")
        except Exception as e:
            logger.error(f"Error during general data preprocessing: {e}")
            raise

    def preprocess_final_eval_data(self):
        """
        Preprocess final evaluation data to prevent data leakage.
        """
        try:
            logger.info("Preprocessing final evaluation data...")
            self.data_preprocessor = DataPreprocessor(
                self.config_instance, self.adata, self.adata_corrected
            )

            batch_correction_enabled = (
                self.config_instance.DataParameters.BatchCorrection.enabled
            )
            adata_eval = (
                self.adata_eval_corrected
                if batch_correction_enabled
                else self.adata_eval
            )
            (
                self.test_data,
                self.test_labels,
                self.label_encoder,
            ) = self.data_preprocessor.prepare_final_eval_data(
                adata_eval,
                self.label_encoder,
                self.num_features,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.mix_included,
            )
            logger.info("Final evaluation data preprocessed successfully.")
        except Exception as e:
            logger.error(f"Error during final evaluation data preprocessing: {e}")
            raise

    def run_preprocessing(self):
        """
        Decides whether to preprocess final evaluation data (if loading model) or
        preprocess training/test data (if building a model).
        """
        try:
            self.setup_gene_filtering()
            if self.config_instance.DataProcessing.ModelManagement.load_model:
                self.load_model_components()
                self.preprocess_final_eval_data()  # Call the final evaluation preprocessing
            else:
                self.preprocess_data()  # Call the general data preprocessing

            logger.info("Preprocessing completed.")

        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise

    def load_model(self):
        """
        Load pre-trained model and preprocess evaluation data to prevent data leakage.
        """
        try:
            # Load the pre-trained model
            self.model = self.model_loader.load_model()

        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            raise

    def load_model_components(self):
        """
        Load pre-trained model and preprocess evaluation data to prevent data leakage.
        """
        try:
            self.model_loader = ModelLoader(self.config_instance)

            (
                self.label_encoder,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.num_features,
                self.history,
                self.mix_included,
                self.reference_data,
            ) = self.model_loader.load_model_components()

        except Exception as e:
            logger.error(f"Error during model loading: {e}")
            raise

    def build_and_train_model(self):
        """
        Preprocess data and build/train a new model.
        """
        try:
            # Build and train the model
            self.model_builder = ModelBuilder(
                self.config_instance,
                self.train_data,
                self.train_labels,
                self.label_encoder,
                self.reference_data,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.mix_included,
            )

            self.model, self.history = self.model_builder.run()

        except Exception as e:
            logger.error(f"Error building or training model: {e}")
            raise

    def load_or_train_model(self):
        """
        Decide whether to load a pre-trained model or build and train a new one.
        """
        try:
            if self.config_instance.DataProcessing.ModelManagement.load_model:
                logger.info("Loading pre-trained model...")
                self.load_model()
                logger.info("Model loaded successfully.")
            else:
                logger.info("Building and training model...")
                self.build_and_train_model()
                logger.info("Model built and trained successfully.")
        except Exception as e:
            logger.error(f"Error in model handling: {e}")
            raise

    def run_interpretation(self):
        """
        Perform SHAP interpretation for model predictions.
        """
        if self.config_instance.FeatureImportanceAndVisualizations.run_interpreter:
            try:
                logger.info("Starting SHAP interpretation...")

                # Initialize the Interpreter with the best model and data
                self.interpreter = Interpreter(
                    self.config_instance,
                    self.model,
                    self.test_data,
                    self.test_labels,
                    self.label_encoder,
                    self.reference_data,
                    self.path_manager,
                )

                # Compute or load SHAP values
                squeezed_shap_values, squeezed_test_data = (
                    self.interpreter.compute_or_load_shap_values()
                )

                # Update SHAP-related attributes for visualization
                self.squeezed_shap_values = squeezed_shap_values
                self.squeezed_test_data = squeezed_test_data

                logger.info("SHAP interpretation complete.")

            except Exception as e:
                logger.error(f"Error during SHAP interpretation: {e}")
                raise
        else:
            self.squeezed_shap_values, self.squeezed_test_data = None, None
            logger.info(
                "Skipping SHAP interpretation as it is disabled in the configuration."
            )

    def run_visualizations(self):
        """
        Set up and run visualizations if specified in the configuration.
        """
        if self.config_instance.FeatureImportanceAndVisualizations.run_visualization:
            logger.info("Running visualizations...")
            visualizer = Visualizer(
                self.config_instance,
                self.model,
                self.history,
                self.test_data,
                self.test_labels,
                self.label_encoder,
                self.squeezed_shap_values,
                self.squeezed_test_data,
                self.adata,
                self.adata_corrected,
                self.path_manager,
            )
            visualizer.run()
            logger.info("Visualizations completed.")
        else:
            logger.info("Visualization is disabled in the configuration.")

    def run_metrics(self):
        """
        Compute and display evaluation metrics for the model.
        """
        try:
            logger.info("Computing evaluation metrics...")
            metrics = Metrics(
                self.config_instance,
                self.model,
                self.test_data,
                self.test_labels,
                self.label_encoder,
                self.path_manager,
            )
            metrics.compute_metrics()
            logger.info("Evaluation metrics computed successfully.")
        except Exception as e:
            logger.error(f"Error computing evaluation metrics: {e}")
            raise

    def display_duration(self, start_time, end_time):
        """
        Displays the duration of a task in a human-readable format.

        Parameters:
        - start_time (float): The start time of the task.
        - end_time (float): The end time of the task.
        """
        duration_seconds = end_time - start_time
        if duration_seconds < 60:
            logger.info(f"The task took {round(duration_seconds)} seconds.")
        else:
            minutes = duration_seconds // 60
            seconds = round(duration_seconds % 60)
            logger.info(f"The task took {int(minutes)} minutes and {seconds} seconds.")
