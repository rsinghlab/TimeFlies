import os
import time
import logging
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Code.config import config
from Code.preprocess import DataPreprocessor, GeneFilter
from Code.utilities import DataLoader, GPUHandler, PathManager, display_duration
from Code.interpreter import Interpreter
from Code.model import ModelBuilder, ModelLoader
from Code.visuals import Visualizer
from Code.eda import EDAHandler

# Set TensorFlow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)


class TimeFlies:
    def __init__(self):
        """
        Initialize the TimeFlies class with configuration and data loader.
        """
        logging.info("Initializing TimeFlies pipeline...")
        self.config_instance = config
        self.data_loader = DataLoader(self.config_instance)
        self.path_manager = PathManager(self.config_instance)

    def setup_gpu(self):
        """
        Configure TensorFlow GPU settings if available.
        """
        try:
            logging.info("Setting up GPU...")
            GPUHandler.configure(self.config_instance)
            logging.info("GPU configuration successful.")
        except Exception as e:
            logging.error(f"Error configuring GPU: {e}")
            raise

    def load_data(self):
        """
        Load data and gene lists.
        """
        try:
            logging.info("Loading data...")
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
            logging.info("Data loading complete.")
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def run_eda(self):
        """
        Perform Exploratory Data Analysis (EDA) if specified in the config.
        """
        try:
            if self.config_instance.DataProcessing.ExploratoryDataAnalysis.enabled:
                logging.info("Running Exploratory Data Analysis (EDA)...")
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
                logging.info("EDA completed.")
            else:
                logging.info("Skipping EDA as it is disabled in the configuration.")
        except Exception as e:
            logging.error(f"Error during EDA: {e}")
            raise

    def setup_gene_filtering(self):
        """
        Filter genes using the GeneFilter class.
        """
        try:
            logging.info("Applying gene filtering...")
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
            logging.info("Gene filtering complete.")
        except Exception as e:
            logging.error(f"Error during gene filtering: {e}")
            raise

    def preprocess_data(self):
        """
        Preprocess general training and test data for building or training a new model.
        """
        try:
            logging.info("Preprocessing data for training and testing...")
            self.data_preprocessor = DataPreprocessor(
                self.config_instance, self.adata, self.adata_eval
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
            logging.info("Training and testing data preprocessed successfully.")
        except Exception as e:
            logging.error(f"Error during general data preprocessing: {e}")
            raise

    def preprocess_final_eval_data(self):
        """
        Preprocess final evaluation data to prevent data leakage.
        """
        try:
            logging.info("Preprocessing final evaluation data...")
            self.data_preprocessor = DataPreprocessor(
                self.config_instance, self.adata, self.adata_eval
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
                self.model,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
            )
            logging.info("Final evaluation data preprocessed successfully.")
        except Exception as e:
            logging.error(f"Error during final evaluation data preprocessing: {e}")
            raise

    def run_preprocessing(self):
        """
        Decides whether to preprocess final evaluation data (if loading model) or
        preprocess training/test data (if building a model).
        """
        try:
            self.setup_gene_filtering()
            if self.config_instance.DataProcessing.ModelManagement.load_model:
                self.preprocess_final_eval_data()  # Call the final evaluation preprocessing
            else:
                self.preprocess_data()  # Call the general data preprocessing

            logging.info("Preprocessing completed.")

        except Exception as e:
            logging.error(f"Error during data preprocessing: {e}")
            raise

    def load_model(self):
        """
        Load pre-trained model and preprocess evaluation data to prevent data leakage.
        """
        try:
            # Load the pre-trained model
            self.model_loader = ModelLoader(self.config_instance)
            (
                self.model,
                self.label_encoder,
                self.reference_data,
                self.scaler,
                self.is_scaler_fit,
                self.mix_included,
                self.highly_variable_genes,
                self.test_data,
                self.test_labels,
                self.history,
            ) = self.model_loader.load_model()

        except Exception as e:
            logging.error(f"Error during model loading: {e}")
            raise

    def load_model_for_shap(self):
        """
        Load pre-trained model and preprocess evaluation data to prevent data leakage.
        """
        try:
            # Load the pre-trained model
            self.model_loader = ModelLoader(self.config_instance)
            (
                self.best_model,
                self.best_label_encoder,
                self.best_reference_data,
                self.best_scaler,
                self.best_is_scaler_fit,
                self.best_mix_included,
                self.best_test_data,
                self.best_test_labels,
                self.best_history,
            ) = self.model_loader.load_model()

        except Exception as e:
            logging.error(f"Error during model loading: {e}")
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
                self.test_data,
                self.test_labels,
                self.label_encoder,
                self.reference_data,
                self.scaler,
                self.is_scaler_fit,
                self.highly_variable_genes,
                self.mix_included,
            )

            self.model, self.history = self.model_builder.run()

        except Exception as e:
            logging.error(f"Error building or training model: {e}")
            raise

    def load_or_train_model(self):
        """
        Decide whether to load a pre-trained model or build and train a new one.
        """
        try:
            if self.config_instance.DataProcessing.ModelManagement.load_model:
                logging.info("Loading pre-trained model...")
                self.load_model()
                logging.info("Model loaded successfully.")
            else:
                logging.info("Building and training model...")
                self.build_and_train_model()
                logging.info("Model built and trained successfully.")
        except Exception as e:
            logging.error(f"Error in model handling: {e}")
            raise

    def run_interpretation(self):
        """
        Perform SHAP interpretation for model predictions.
        """
        if self.config_instance.FeatureImportanceAndVisualizations.run_interpreter:
            try:
                logging.info("Starting SHAP interpretation...")

                # Load the best model and data without overwriting current data
                self.load_model_for_shap()

                # Initialize the Interpreter with the best model and data
                self.interpreter = Interpreter(
                    self.config_instance,
                    self.best_model,
                    self.best_test_data,
                    self.best_test_labels,
                    self.best_label_encoder,
                    self.best_reference_data,
                    self.path_manager,
                )

                # Compute or load SHAP values
                shap_values, squeezed_test_data = (
                    self.interpreter.compute_or_load_shap_values()
                )

                # Update SHAP-related attributes for visualization
                self.squeezed_shap_values = shap_values
                self.squeezed_test_data = squeezed_test_data

                logging.info("SHAP interpretation complete.")

            except Exception as e:
                logging.error(f"Error during SHAP interpretation: {e}")
                raise
        else:
            self.squeezed_shap_values, self.squeezed_test_data = None, None
            logging.info(
                "Skipping SHAP interpretation as it is disabled in the configuration."
            )

    def run_visualizations(self):
        """
        Set up and run visualizations if specified in the configuration.
        """
        if self.config_instance.FeatureImportanceAndVisualizations.run_visualization:
            logging.info("Running visualizations...")
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
            logging.info("Visualizations completed.")
        else:
            logging.info("Visualization is disabled in the configuration.")

    def run(self):
        """
        Main pipeline to orchestrate the entire workflow.
        """
        start_time = time.time()

        try:
            # Step 1: Configure GPU
            self.setup_gpu()

            # Step 2: Data loading
            self.load_data()

            # Step 3: Run EDA if configured
            self.run_eda()

            # Step 4: Model handling and preprocessing
            if self.config_instance.DataProcessing.ModelManagement.load_model:
                # If loading a model, load it before preprocessing
                self.load_or_train_model()
            # Preprocess data based on whether a model is loaded or not
            self.run_preprocessing()

            if not self.config_instance.DataProcessing.ModelManagement.load_model:
                # If not loading a model, build and train after preprocessing
                self.load_or_train_model()

            # Step 6: Perform interpretation and visualization
            self.run_interpretation()
            self.run_visualizations()

            # Step 7: Display duration
            end_time = time.time()
            display_duration(start_time, end_time)

        except Exception as e:
            logging.error(f"Error during pipeline execution: {e}")
            raise


if __name__ == "__main__":
    pipeline = TimeFlies()
    pipeline.run()
