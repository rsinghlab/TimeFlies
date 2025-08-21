import logging
from typing import Optional, Dict, Any

from .config_manager import Config
from shared.data.preprocessing.data_processor import DataPreprocessor
from shared.data.preprocessing.gene_filter import GeneFilter
from shared.models.model import ModelBuilder, ModelLoader
from shared.utils.gpu_handler import GPUHandler
from shared.utils.path_manager import PathManager
from shared.data.loaders import DataLoader

# Optional project-specific components (injected at runtime)
EDAHandler = None
Interpreter = None  
Metrics = None
Visualizer = None

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

    def __init__(self, config: Config):
        """
        Initialize the PipelineManager class with configuration and data loader.

        Args:
            config: Configuration object
        """
        self.config_instance = config
        self.data_loader = DataLoader(self.config_instance)
        self.path_manager = PathManager(self.config_instance)
        
        # Auto-inject shared analysis components
        try:
            from shared.analysis import EDAHandler, Visualizer
            from shared.evaluation import Interpreter, Metrics
            self.eda_handler_class = EDAHandler
            self.interpreter_class = Interpreter
            self.metrics_class = Metrics
            self.visualizer_class = Visualizer
        except ImportError:
            self.eda_handler_class = None
            self.interpreter_class = None
            self.metrics_class = None
            self.visualizer_class = None
        
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
            # Only load batch-corrected data if batch correction is enabled
            batch_correction_enabled = getattr(
                self.config_instance.data.batch_correction, "enabled", False
            )
            if batch_correction_enabled:
                logger.info("Batch correction enabled - loading corrected data...")
                (
                    self.adata_corrected,
                    self.adata_eval_corrected,
                ) = self.data_loader.load_corrected_data()
            else:
                logger.info(
                    "Batch correction disabled - skipping corrected data loading"
                )
                self.adata_corrected = None
                self.adata_eval_corrected = None
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
            if getattr(
                self.config_instance.data_processing, "exploratory_data_analysis", {}
            ).get("enabled", False):
                logger.info("Running Exploratory Data Analysis (EDA)...")
                if self.eda_handler_class is not None:
                    eda_handler = self.eda_handler_class(
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
                    logger.warning("EDA requested but no EDA handler available")
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

            # Free memory by deleting large raw data objects after preprocessing
            # (only during training - evaluation uses dedicated run_evaluation method)
            logger.info("Cleaning up raw data objects to free memory...")
            del self.adata
            del self.adata_eval
            del self.adata_original
            if (
                hasattr(self, "adata_corrected")
                and self.adata_corrected is not None
            ):
                del self.adata_corrected
            if (
                hasattr(self, "adata_eval_corrected")
                and self.adata_eval_corrected is not None
            ):
                del self.adata_eval_corrected

            import gc

            gc.collect()  # Force garbage collection
            logger.info("Memory cleanup complete.")
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

            batch_correction_enabled = getattr(
                self.config_instance.data.batch_correction, "enabled", False
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

            # Preserve gene names before memory cleanup for visualization
            logger.info("Preserving gene names for visualization...")
            batch_correction_enabled = getattr(
                self.config_instance.data.batch_correction, "enabled", False
            )
            select_batch_genes = getattr(
                self.config_instance.preprocessing.genes,
                "select_batch_genes",
                False,
            )

            if batch_correction_enabled or select_batch_genes:
                if (
                    hasattr(self, "adata_corrected")
                    and self.adata_corrected is not None
                ):
                    self.preserved_var_names = self.adata_corrected.var_names.copy()
                elif (
                    hasattr(self, "adata_eval_corrected")
                    and self.adata_eval_corrected is not None
                ):
                    self.preserved_var_names = (
                        self.adata_eval_corrected.var_names.copy()
                    )
                else:
                    self.preserved_var_names = getattr(
                        self, "adata", getattr(self, "adata_eval", None)
                    )
                    if self.preserved_var_names is not None:
                        self.preserved_var_names = (
                            self.preserved_var_names.var_names.copy()
                        )
            else:
                if hasattr(self, "adata"):
                    self.preserved_var_names = self.adata.var_names.copy()
                elif hasattr(self, "adata_eval"):
                    self.preserved_var_names = self.adata_eval.var_names.copy()

            # Free memory by deleting large raw data objects after final evaluation preprocessing
            logger.info("Cleaning up raw data objects to free memory...")
            if hasattr(self, "adata"):
                del self.adata
            if hasattr(self, "adata_eval"):
                del self.adata_eval
            if hasattr(self, "adata_original"):
                del self.adata_original
            if hasattr(self, "adata_corrected") and self.adata_corrected is not None:
                del self.adata_corrected
            if (
                hasattr(self, "adata_eval_corrected")
                and self.adata_eval_corrected is not None
            ):
                del self.adata_eval_corrected

            import gc

            gc.collect()  # Force garbage collection
            logger.info("Memory cleanup complete.")
        except Exception as e:
            logger.error(f"Error during final evaluation data preprocessing: {e}")
            raise

    def run_preprocessing(self):
        """
        Preprocess training/test data for model building.
        Note: Evaluation uses the dedicated run_evaluation() method instead.
        """
        try:
            self.setup_gene_filtering()
            self.preprocess_data()  # Always do training preprocessing
            logger.info("Preprocessing completed.")

        except Exception as e:
            logger.error(f"Error during data preprocessing: {e}")
            raise

    def load_model(self):
        """
        Load pre-trained model and preprocess evaluation data to prevent data leakage.
        """
        try:
            logger.info("Loading pre-trained model...")
            # Initialize model loader if not already done
            if not hasattr(self, "model_loader"):
                self.model_loader = ModelLoader(self.config_instance)

            # Load the pre-trained model
            self.model = self.model_loader.load_model()
            logger.info("Pre-trained model loaded successfully.")

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
            print("History keys:", self.history.history.keys())

        except Exception as e:
            logger.error(f"Error building or training model: {e}")
            raise

    def load_or_train_model(self):
        """
        Build and train a new model (training pipeline only).
        Note: Model loading is handled by the dedicated run_evaluation() method.
        """
        try:
            logger.info("Building and training model...")
            self.build_and_train_model()
            logger.info("Model built and trained successfully.")
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def run_interpretation(self):
        """
        Perform SHAP interpretation for model predictions.
        """
        if getattr(self.config_instance.interpretation.shap, "enabled", False):
            try:
                logger.info("Starting SHAP interpretation...")
                if self.interpreter_class is None:
                    logger.warning("Interpreter class not available, skipping SHAP interpretation")
                    return

                # Initialize the Interpreter with the best model and data
                self.interpreter = self.interpreter_class(
                    self.config_instance,
                    self.model,
                    self.test_data,
                    self.test_labels,
                    self.label_encoder,
                    self.reference_data,
                    self.path_manager,
                )

                # Compute or load SHAP values
                (
                    squeezed_shap_values,
                    squeezed_test_data,
                ) = self.interpreter.compute_or_load_shap_values()

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
        if getattr(self.config_instance.visualizations, "enabled", False):
            logger.info("Running visualizations...")

            # Ensure SHAP attributes exist (set to None if not available)
            if not hasattr(self, "squeezed_shap_values"):
                self.squeezed_shap_values = None
            if not hasattr(self, "squeezed_test_data"):
                self.squeezed_test_data = None

            # Ensure adata attributes exist (may be deleted for memory cleanup)
            adata = getattr(self, "adata", None)
            adata_corrected = getattr(self, "adata_corrected", None)

            if self.visualizer_class is None:
                logger.warning("Visualizer class not available, skipping visualizations")
                return
                
            visualizer = self.visualizer_class(
                self.config_instance,
                self.model,
                self.history,
                self.test_data,
                self.test_labels,
                self.label_encoder,
                self.squeezed_shap_values,
                self.squeezed_test_data,
                adata,
                adata_corrected,
                self.path_manager,
                preserved_var_names=getattr(self, "preserved_var_names", None),
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
            if self.metrics_class is None:
                logger.warning("Metrics class not available, skipping metrics computation")
                return
                
            metrics = self.metrics_class(
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

    def run_shap_interpretation(self):
        """Run SHAP interpretation if enabled in config."""
        try:
            logger.info("Running SHAP interpretation...")
            from ..evaluation.interpreter import Interpreter

            interpreter = Interpreter(
                self.config_instance,
                self.model,
                self.test_data,
                self.test_labels,
                self.path_manager,
            )
            interpreter.run_interpretation()
            logger.info("SHAP interpretation completed successfully.")
        except Exception as e:
            logger.error(f"Error running SHAP interpretation: {e}")
            raise

    def run_analysis_script(self):
        """
        Run project-specific analysis script if available.
        """
        try:
            logger.info("Running project-specific analysis script...")
            
            # Determine project name and import the appropriate analyzer
            project_name = getattr(self.config_instance, "project", "fruitfly_aging")
            
            if project_name == "fruitfly_alzheimers":
                try:
                    from projects.fruitfly_alzheimers.analysis.aging_acceleration_analyzer import AlzheimersAgingAccelerationAnalyzer
                    
                    # Create analyzer with optional model and data
                    analyzer = AlzheimersAgingAccelerationAnalyzer(
                        model=getattr(self, 'model', None),
                        config=self.config_instance,
                        path_manager=getattr(self, 'path_manager', None)
                    )
                    
                    # Run the analysis
                    analyzer.run_complete_analysis()
                    logger.info("Alzheimer's aging acceleration analysis completed.")
                    
                except ImportError:
                    logger.warning("Alzheimer's analysis script not found, skipping analysis.")
                    
            elif project_name == "fruitfly_aging":
                try:
                    from projects.fruitfly_aging.analysis.analyzer import AgingAnalyzer
                    
                    # Create analyzer (would need to be implemented)
                    logger.info("Aging analysis script would run here (not yet implemented).")
                    
                except ImportError:
                    logger.warning("Aging analysis script not found, skipping analysis.")
            else:
                logger.info(f"No analysis script configured for project: {project_name}")
                
        except Exception as e:
            logger.error(f"Error running analysis script: {e}")
            # Don't raise - analysis scripts are optional

    def run_evaluation(self) -> Dict[str, Any]:
        """
        Run evaluation-only pipeline that uses pre-trained model on holdout data.
        
        Returns:
            Dict containing model path and results path
        """
        import time
        
        start_time = time.time()
        
        try:
            # Setup phases
            self.setup_gpu()
            self.load_data()
            self.setup_gene_filtering()
            
            # Load model components and preprocess only eval data
            self.load_model_components()
            self.preprocess_final_eval_data()
            
            # Load pre-trained model
            self.load_model()
            
            # Run evaluation metrics
            self.run_metrics()
            
            # Analysis (conditional based on config)
            if getattr(self.config_instance.interpretation.shap, "enabled", False):
                self.run_interpretation()
                
            if getattr(self.config_instance.visualizations, "enabled", False):
                self.run_visualizations()
            
            # Run project-specific analysis script if enabled
            if getattr(
                self.config_instance.analysis.run_analysis_script, "enabled", False
            ):
                self.run_analysis_script()
            
            end_time = time.time()
            self.display_duration(start_time, end_time)
            
            # Return paths for CLI feedback
            model_dir = self.path_manager.construct_model_directory()
            results_dir = self.path_manager.get_visualization_directory()
            
            return {
                "model_path": model_dir,
                "results_path": results_dir,
                "duration": end_time - start_time,
            }
            
        except Exception as e:
            logger.error(f"Evaluation pipeline execution failed: {e}")
            raise

    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline and return results.

        Returns:
            Dict containing model path and results path
        """
        import time

        start_time = time.time()

        try:
            # Setup phases
            self.setup_gpu()
            self.load_data()
            self.setup_gene_filtering()

            # Preprocessing
            self.preprocess_data()

            # Model training/loading
            self.load_or_train_model()

            # No evaluation during training - that's handled by the evaluate command
            self.run_metrics()

            # Analysis (conditional based on config)
            if getattr(
                self.config_instance.interpretation.shap, "enabled", False
            ):
                self.run_shap_interpretation()

            if getattr(
                self.config_instance.visualizations, "enabled", False
            ):
                self.run_visualizations()

            # Run project-specific analysis script if enabled
            if getattr(
                self.config_instance.analysis.run_analysis_script, "enabled", False
            ):
                self.run_analysis_script()

            end_time = time.time()
            self.display_duration(start_time, end_time)

            # Return paths for CLI feedback
            model_dir = self.path_manager.construct_model_directory()
            results_dir = self.path_manager.get_visualization_directory()

            return {
                "model_path": model_dir,
                "results_path": results_dir,
                "duration": end_time - start_time,
            }

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise
