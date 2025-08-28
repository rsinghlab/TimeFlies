"""Data processing and display management for pipeline operations."""
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DataProcessingManager:
    """Handles complex data processing display and evaluation data preparation."""

    def __init__(self, config, display_utils, constants_class):
        self.config = config
        self.display_utils = display_utils
        self.constants = constants_class

    def _get_display_columns(self):
        """
        Get the list of columns to display in distributions.

        This method dynamically builds the column list from:
        1. User-specified columns in config.data.display_columns
        2. Automatically adds cell type column if cell filtering is enabled

        This allows flexible display of any columns (sex, genotype, age, batch, etc.)
        without hardcoding specific column names.
        """
        display_columns = list(getattr(self.config.data, "display_columns", ["sex", "genotype"]))

        # Automatically add cell type column if cell filtering is enabled
        cell_filtering = getattr(self.config.data, "cell_filtering", None)
        if cell_filtering:
            cell_type_column = getattr(cell_filtering, "column", None)
            if cell_type_column and cell_type_column not in display_columns:
                display_columns.append(cell_type_column)
        else:
            # Fallback for backwards compatibility
            cell_type_column = getattr(self.config.data, "cell_type_column", None)
            if cell_type_column and cell_type_column not in display_columns:
                display_columns.append(cell_type_column)

        return display_columns

    def display_preprocessed_data_overview(self, pipeline_obj):
        """Display training and evaluation data details after preprocessing."""
        print("\n")
        print("PREPROCESSED DATA OVERVIEW")
        self.display_utils.print_section_separator()

        # Check if we have any preprocessed data to display
        if not (hasattr(pipeline_obj, "train_data") or hasattr(pipeline_obj, "test_data")):
            print("No preprocessed data available to display.")
            return

        # Training data details
        self._display_training_data(pipeline_obj)

        # Always show processed evaluation data (the proper eval dataset)
        # This uses the correctly processed evaluation data regardless of split method
        self._show_processed_eval_data_preview(pipeline_obj)

        # Display split configuration
        self._display_split_configuration()

    def _display_training_data(self, pipeline_obj):
        """Display training data statistics and distributions."""
        if not (hasattr(pipeline_obj, "train_data") and pipeline_obj.train_data is not None):
            print("Training Data: Not available")
            return

        try:
            print("Training Data:")
            self.display_utils.print_shape_info(pipeline_obj.train_data)
            self.display_utils.print_data_statistics(pipeline_obj.train_data)

            # Training labels distribution
            if hasattr(pipeline_obj, "train_labels") and pipeline_obj.train_labels is not None:
                encoding_var = getattr(self.config.data, "target_variable", self.constants.DEFAULT_TARGET_VARIABLE)
                label_encoder = getattr(pipeline_obj, "label_encoder", None)
                self.display_utils.print_label_distribution(
                    pipeline_obj.train_labels,
                    label_encoder,
                    f"  └─ {encoding_var.title()} Distribution"
                )

            # Additional distributions if available
            if hasattr(pipeline_obj, 'train_subset') and pipeline_obj.train_subset is not None:
                display_columns = self._get_display_columns()
                self.display_utils.print_obs_distributions(
                    pipeline_obj.train_subset,
                    display_columns
                )

        except Exception as e:
            print(f"Training Data: Could not display details ({e})")

    def _display_validation_data(self, pipeline_obj):
        """Display validation/test data from train/validation split."""
        if not (hasattr(pipeline_obj, "test_data") and pipeline_obj.test_data is not None):
            print("\nValidation Data: Not available")
            return

        try:
            test_shape = pipeline_obj.test_data.shape
            if test_shape[0] > 0:  # Only show if we have samples
                print("\nValidation Data:")
                self.display_utils.print_shape_info(pipeline_obj.test_data)
                self.display_utils.print_data_statistics(pipeline_obj.test_data)

                # Test labels distribution
                if hasattr(pipeline_obj, "test_labels") and pipeline_obj.test_labels is not None:
                    encoding_var = getattr(self.config.data, "target_variable", self.constants.DEFAULT_TARGET_VARIABLE)
                    label_encoder = getattr(pipeline_obj, "label_encoder", None)
                    self.display_utils.print_label_distribution(
                        pipeline_obj.test_labels,
                        label_encoder,
                        f"\nValidation {encoding_var.title()} Distribution"
                    )

        except Exception as e:
            print(f"\nValidation Data: Could not display details ({e})")



    def _show_fallback_eval_info(self, pipeline_obj):
        """Show fallback evaluation data info if processing fails."""
        try:
            adata_eval = getattr(pipeline_obj, "adata_eval_corrected", None) or getattr(
                pipeline_obj, "adata_eval", None
            )
            if adata_eval is not None:
                eval_cells = adata_eval.n_obs
                eval_genes = adata_eval.n_vars
                print(f"  └─ Raw Samples:       {eval_cells:,}")
                print(f"  └─ Raw Features:      {eval_genes:,}")
            else:
                print("  └─ No evaluation data available")
        except Exception as fallback_e:
            print(f"  └─ Could not show fallback info: {fallback_e}")

    def _display_split_configuration(self):
        """Display split configuration details."""
        try:
            from common.utils.split_naming import SplitNamingUtils
            split_config = SplitNamingUtils.extract_split_details_for_metadata(self.config)
            self.display_utils.print_split_configuration(split_config)
        except Exception as e:
            logger.warning(f"Could not display split configuration: {e}")
