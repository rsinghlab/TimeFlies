import os
import time
import logging

from pipeline_manager import PipelineManager

# Set TensorFlow log level
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Set up logging
logging.basicConfig(level=logging.INFO)


class TimeFlies:
    """
    The TimeFlies class serves as the main entry point for the application.
    It initializes the pipeline and orchestrates the execution by calling
    the appropriate methods in the PipelineManager.

    Attributes:
        pipeline_manager (PipelineManager): An instance of PipelineManager to manage the pipeline steps.
    """

    def __init__(self):
        """
        Initialize the TimeFlies class with PipelineManager.
        """
        logging.info("Initializing TimeFlies pipeline...")
        self.pipeline_manager = PipelineManager()
    def run(self):
        """
        Main pipeline to orchestrate the entire workflow.
        """
        pipeline_start_time = time.time()  # Track entire pipeline start time

        try:
            # Step 1: Configure GPU
            self.pipeline_manager.setup_gpu()

            # Step 2: Data loading
            self.pipeline_manager.load_data()

            # Step 3: Run EDA if configured
            self.pipeline_manager.run_eda()

            # Step 4: Preprocessing
            self.pipeline_manager.run_preprocessing()

            # Step 5: Model handling
            self.pipeline_manager.load_or_train_model()

            # Step 6: Perform interpretation, visualization, and metric calculations
            self.pipeline_manager.run_interpretation()
            self.pipeline_manager.run_visualizations()
            self.pipeline_manager.run_metrics()

            # Step 7: Display pipeline duration
            end_time = time.time()
            self.pipeline_manager.display_duration(pipeline_start_time, end_time)

        except Exception as e:
            logging.error(f"Error during pipeline execution: {e}")
            raise

if __name__ == "__main__":
    pipeline = TimeFlies()
    pipeline.run()
