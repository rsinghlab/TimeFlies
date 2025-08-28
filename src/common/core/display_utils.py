"""Utilities for displaying pipeline information and data statistics."""
import logging
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class PipelineDisplayUtils:
    """Utility class for formatting and displaying pipeline information."""

    def __init__(self, config, constants_class):
        self.config = config
        self.constants = constants_class

    def print_header(self, title: str, width: int = None):
        """Print a formatted header with title."""
        if width is None:
            width = self.constants.HEADER_WIDTH
        print("\n" + "=" * width)
        print(title)
        print("=" * width)

    def print_section_separator(self):
        """Print section separator line."""
        print(self.constants.SECTION_SEPARATOR)

    def print_subsection_separator(self):
        """Print subsection separator line."""
        print(self.constants.SUBSECTION_SEPARATOR)

    def print_distribution(self, title: str, distribution_dict: dict[Any, int],
                          total: int, indent: str = "  └─"):
        """Print a distribution with percentages."""
        print(f"{title}:")
        for key, count in distribution_dict.items():
            pct = (count / total) * 100
            print(f"{indent} {key:<{self.constants.COLUMN_WIDTH}}: {count:6,} samples ({pct:5.{self.constants.PERCENTAGE_PRECISION}f}%)")

    def print_data_statistics(self, data: np.ndarray, prefix: str = "  └─"):
        """Print standard data statistics (mean, std, range)."""
        data_mean = np.mean(data)
        data_std = np.std(data)
        data_min = np.min(data)
        data_max = np.max(data)
        print(f"{prefix} Data Range:        [{data_min:.{self.constants.VALUE_PRECISION}f}, {data_max:.{self.constants.VALUE_PRECISION}f}]")
        print(f"{prefix} Mean ± Std:        {data_mean:.{self.constants.VALUE_PRECISION}f} ± {data_std:.{self.constants.VALUE_PRECISION}f}")

    def print_shape_info(self, data: np.ndarray, data_type: str = "Data", prefix: str = "  └─"):
        """Print shape information for data arrays."""
        shape = data.shape
        print(f"{prefix} Samples:           {shape[0]:,}")
        if len(shape) > 2:  # CNN format
            print(f"{prefix} Features:          {shape[1]} x {shape[2]:,} (reshaped)")
        else:  # Standard format
            print(f"{prefix} Features (genes):  {shape[1]:,}")

    def print_dataset_overview(self, adata, dataset_name: str):
        """Print overview of a dataset."""
        if adata is None:
            return

        cells = adata.n_obs
        genes = adata.n_vars
        print(f"{dataset_name} Dataset Size:")
        print(f"  └─ Samples:           {cells:,}")
        print(f"  └─ Features (genes):  {genes:,}")

        # Show target distribution
        target = getattr(self.config.data, "target_variable", self.constants.DEFAULT_TARGET_VARIABLE)
        target_name = target if isinstance(target, str) else str(target)

        if target in adata.obs.columns:
            print(f"\n{dataset_name} {target_name.title()} Distribution:")
            dist = adata.obs[target].value_counts().sort_index()
            total = dist.sum()
            self.print_distribution("", dist.to_dict(), total)

    def print_label_distribution(self, labels: np.ndarray, label_encoder=None,
                                title: str = None, prefix: str = "  └─"):
        """Print distribution of labels with proper decoding."""
        try:
            # Handle both one-hot encoded and label encoded data
            if hasattr(labels, "shape") and len(labels.shape) > 1 and labels.shape[1] > 1:
                # One-hot encoded - convert to class indices
                label_indices = np.argmax(labels, axis=1)
            else:
                # Already class indices or 1D array
                label_indices = np.array(labels).flatten()

            unique, counts = np.unique(label_indices, return_counts=True)
            total = counts.sum()

            if title:
                print(f"{title}:")

            for label_encoded, count in zip(unique, counts):
                pct = (count / total) * 100
                try:
                    if label_encoder is not None:
                        original_label = label_encoder.inverse_transform([int(label_encoded)])[0]
                        print(f"{prefix}   └─ {original_label:<{self.constants.COLUMN_WIDTH}}: {count:6,} samples ({pct:5.{self.constants.PERCENTAGE_PRECISION}f}%)")
                    else:
                        print(f"{prefix}   └─ {label_encoded:<{self.constants.COLUMN_WIDTH}}: {count:6,} samples ({pct:5.{self.constants.PERCENTAGE_PRECISION}f}%)")
                except Exception:
                    # Fallback to showing encoded label
                    print(f"{prefix}   └─ Class {label_encoded:<7}: {count:6,} samples ({pct:5.{self.constants.PERCENTAGE_PRECISION}f}%)")

        except Exception as e:
            print(f"{prefix}   └─ Could not display label distribution: {e}")

    def print_obs_distributions(self, adata, columns: list[str]):
        """Print distributions for specified obs columns."""
        for col in columns:
            if col in adata.obs.columns:
                print(f"  └─ {col.replace('_', ' ').title()} Distribution:")
                counts = adata.obs[col].value_counts().sort_index()
                for value, count in counts.items():
                    percentage = (count / len(adata.obs)) * 100
                    print(f"      └─ {value:<{self.constants.COLUMN_WIDTH}}: {count:6,} samples ({percentage:5.{self.constants.PERCENTAGE_PRECISION}f}%)")

    def print_split_configuration(self, split_config: dict[str, Any]):
        """Print split configuration details."""
        if not split_config:
            return

        print("\nSplit Configuration:")
        print(f"  └─ Split Method:      {split_config.get('method', 'unknown').title()}")

        if split_config.get("method") == "column":
            print(f"  └─ Split Column:      {split_config.get('column', 'unknown')}")
            train_vals = split_config.get("train_values", [])
            test_vals = split_config.get("test_values", [])
            if train_vals:
                print(f"  └─ Training Values:   {', '.join(train_vals)}")
            if test_vals:
                print(f"  └─ Test Values:       {', '.join(test_vals)}")

    def print_timing_summary(self, preprocessing_duration: float, training_duration: float,
                           evaluation_duration: float):
        """Print timing information summary."""
        print()
        print("TIMING")

        # Calculate actual training time (excluding preprocessing)
        actual_training_duration = training_duration - preprocessing_duration
        total_time = training_duration + evaluation_duration

        print(f"  └─ Preprocessing Duration:    {preprocessing_duration:.1f} seconds")
        print(f"  └─ Model Training Duration:   {actual_training_duration:.1f} seconds")
        print(f"  └─ Evaluation Duration:       {evaluation_duration:.1f} seconds")
        print(f"  └─ Total Time:                {total_time:.1f} seconds")
