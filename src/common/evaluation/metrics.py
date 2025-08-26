"""
Aging-Specific Metrics Module

Specialized evaluation metrics for aging research in Drosophila.
"""

from typing import Any

import numpy as np
import scipy.stats as stats
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from common.utils.logging_config import get_logger

logger = get_logger(__name__)


class EvaluationMetrics:
    """
    General metrics for machine learning model evaluation.

    Provides methods for:
    - Regression metrics (MAE, RMSE, R², correlation)
    - Classification metrics (accuracy, precision, recall)
    - Prediction analysis and visualization
    - Cross-validation and statistical testing
    """

    def __init__(
        self,
        config=None,
        model=None,
        test_data=None,
        test_labels=None,
        label_encoder=None,
        path_manager=None,
        result_type="recent",
        output_dir=None,
    ):
        """
        Initialize aging metrics calculator.

        Args:
            config: Configuration object with metric parameters
            model: Trained model for predictions
            test_data: Test data for evaluation
            test_labels: Test labels/targets
            label_encoder: Label encoder for predictions
            path_manager: Path manager for saving results
            result_type: "recent" (standalone) or "best" (post-training)
        """
        self.config = config
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.path_manager = path_manager
        self.result_type = result_type
        self.output_dir = output_dir

    def compute_metrics(self):
        """
        Compute and save metrics using the model and test data.

        This method is called by the pipeline to evaluate model performance.
        """
        if self.model is None or self.test_data is None or self.test_labels is None:
            logger.warning("Missing model or test data, skipping metrics computation")
            return

        # Make predictions
        predictions = self.model.predict(self.test_data, verbose=0)

        # For classification, get predicted classes
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predicted_classes = np.argmax(predictions, axis=1)
            if self.label_encoder:
                predicted_classes = self.label_encoder.inverse_transform(
                    predicted_classes
                )
        else:
            predicted_classes = predictions.flatten()

        # Get true labels - extract the actual values
        true_labels = self.test_labels
        if hasattr(self.test_labels, "values"):
            true_labels = self.test_labels.values
        elif hasattr(self.test_labels, "to_numpy"):
            true_labels = self.test_labels.to_numpy()
        elif hasattr(self.test_labels, "__array__"):
            true_labels = np.array(self.test_labels)

        # If true_labels is 2D (one-hot encoded), get the class indices
        if len(true_labels.shape) > 1 and true_labels.shape[1] > 1:
            true_labels = np.argmax(true_labels, axis=1)
            if self.label_encoder:
                true_labels = self.label_encoder.inverse_transform(true_labels)

        # Handle categorical true labels - convert to numeric if needed
        if hasattr(true_labels, "dtype") and true_labels.dtype == "object":
            # For age data, convert string ages to numeric
            try:
                true_labels = true_labels.astype(float)
            except (ValueError, TypeError):
                pass

        # Ensure both arrays are numeric and same type
        true_labels = np.asarray(true_labels, dtype=float).flatten()
        predicted_classes = np.asarray(predicted_classes, dtype=float).flatten()

        # Debug logging removed for cleaner output

        # Compute metrics based on task type
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # Classification task - compute both classification and regression metrics
            metrics = self.evaluate_classification_and_regression(
                true_labels, predicted_classes, predictions
            )
        else:
            # Pure regression task
            metrics = self.evaluate_age_prediction(true_labels, predicted_classes)

        # Save metrics if path manager is available
        if self.path_manager:
            import json
            import os

            # Use experiment-specific directory or fallback to global
            if self.output_dir:
                metrics_dir = self.output_dir
                metrics_filename = "metrics.json"
            else:
                # Fallback to old global location
                output_root = self.path_manager.get_outputs_directory()
                metrics_dir = os.path.join(output_root, "metrics")
                metrics_filename = "evaluation_metrics.json"

            # Create directory if it doesn't exist (skip during tests)
            if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
                os.makedirs(metrics_dir, exist_ok=True)

            # Save metrics to JSON
            metrics_file = os.path.join(metrics_dir, metrics_filename)
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)

            # Metrics saved

        # Save predictions to CSV for analysis scripts
        if self.path_manager:
            import pandas as pd

            # Save predictions to experiment directory or fallback to old structure
            if self.output_dir:
                results_dir = self.output_dir
            else:
                # Fallback to old structure
                results_dir = self.path_manager.get_results_dir(
                    self.result_type, "eval"
                )

            # Create directory if it doesn't exist (skip during tests)
            if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI")):
                os.makedirs(results_dir, exist_ok=True)

            # Create predictions dataframe
            predictions_df = pd.DataFrame(
                {
                    "actual_age": true_labels,
                    "predicted_age": predicted_classes,
                    "prediction_error": predicted_classes - true_labels,
                }
            )

            # Add genotype information - try to get actual genotypes from evaluation data
            project = getattr(self.config, "project", "unknown")
            split_method = getattr(self.config.data.split, "method", "random")

            # Try to load actual genotype information from evaluation data
            genotype_info = self._get_evaluation_genotypes(project)

            if genotype_info is not None and len(genotype_info) == len(predictions_df):
                # Use actual genotype information
                predictions_df["genotype"] = genotype_info
            elif project == "fruitfly_alzheimers" and split_method == "genotype":
                # Fallback: Test set is Alzheimer's flies (AB42/hTau)
                predictions_df["genotype"] = "alzheimers"
            elif project == "fruitfly_alzheimers":
                # Fallback: For Alzheimer's project with other split methods
                predictions_df["genotype"] = "alzheimers"
            else:
                predictions_df["genotype"] = "control"

            # Save to CSV
            predictions_file = os.path.join(results_dir, "predictions.csv")
            predictions_df.to_csv(predictions_file, index=False)
            # Predictions saved

        # Only log regression metrics if they exist (for regression tasks)
        mae = metrics.get("mae", None)
        rmse = metrics.get("rmse", None)
        r2 = metrics.get("r2_score", None)
        pearson = metrics.get("pearson_correlation", None)

        if any([mae, rmse, r2, pearson]):
            logger.info("📊 Regression Results:")
            if mae is not None:
                logger.info(f"  MAE: {mae:.4f}")
            if rmse is not None:
                logger.info(f"  RMSE: {rmse:.4f}")
            if r2 is not None:
                logger.info(f"  R² Score: {r2:.4f}")
            if pearson is not None:
                logger.info(f"  Pearson Correlation: {pearson:.4f}")

        return metrics

    def _get_evaluation_genotypes(self, project):
        """
        Load actual genotype information from evaluation data.

        Args:
            project: Project name (e.g., "fruitfly_alzheimers")

        Returns:
            List of genotype labels matching the evaluation samples, or None if not available
        """
        try:
            from common.data.loaders import DataLoader

            # Load evaluation data to get genotype information
            data_loader = DataLoader(self.config)
            _, adata_eval, _ = data_loader.load_data()

            # Check if genotype column exists
            if "genotype" not in adata_eval.obs.columns:
                return None

            # Apply same filtering that was used during evaluation
            split_config = getattr(self.config.data, "split", None)
            if split_config and hasattr(split_config, "test"):
                test_genotypes = getattr(split_config, "test", [])
                if test_genotypes:
                    # Filter to only test genotypes
                    mask = adata_eval.obs["genotype"].isin(test_genotypes)
                    filtered_genotypes = adata_eval.obs["genotype"][mask].tolist()
                    return filtered_genotypes

            # Fallback: return all genotypes
            return adata_eval.obs["genotype"].tolist()

        except Exception:
            # If we can't load genotype info, return None to use fallback
            return None

    def evaluate_age_prediction(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate age prediction performance with aging-specific metrics.

        Args:
            y_true: True age values
            y_pred: Predicted age values

        Returns:
            Dictionary containing various age prediction metrics
        """
        metrics = {}

        # Standard regression metrics
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["r2_score"] = r2_score(y_true, y_pred)

        # Aging-specific metrics
        metrics["mean_age_error_days"] = np.mean(np.abs(y_true - y_pred))
        metrics["median_age_error_days"] = np.median(np.abs(y_true - y_pred))

        # Age-specific accuracy (within X days)
        for threshold in [5, 10, 15, 20]:
            accuracy = np.mean(np.abs(y_true - y_pred) <= threshold)
            metrics[f"accuracy_within_{threshold}_days"] = accuracy

        # Correlation metrics
        pearson_r, pearson_p = stats.pearsonr(y_true, y_pred)
        spearman_r, spearman_p = stats.spearmanr(y_true, y_pred)

        metrics["pearson_correlation"] = pearson_r
        metrics["pearson_p_value"] = pearson_p
        metrics["spearman_correlation"] = spearman_r
        metrics["spearman_p_value"] = spearman_p

        # Age group consistency
        age_group_metrics = self._evaluate_age_group_consistency(y_true, y_pred)
        metrics.update(age_group_metrics)

        return metrics

    def evaluate_sex_specific_performance(
        self, y_true: np.ndarray, y_pred: np.ndarray, sex_labels: np.ndarray
    ) -> dict[str, Any]:
        """
        Evaluate performance separately for males and females.

        Args:
            y_true: True age values
            y_pred: Predicted age values
            sex_labels: Sex labels ('male' or 'female')

        Returns:
            Dictionary with sex-specific performance metrics
        """
        results = {}

        for sex in ["male", "female"]:
            sex_mask = sex_labels == sex
            if np.sum(sex_mask) == 0:
                continue

            sex_true = y_true[sex_mask]
            sex_pred = y_pred[sex_mask]

            sex_metrics = self.evaluate_age_prediction(sex_true, sex_pred)
            results[f"{sex}_metrics"] = sex_metrics

        # Compare performance between sexes
        if "male_metrics" in results and "female_metrics" in results:
            results["sex_comparison"] = {
                "mae_difference": results["male_metrics"]["mae"]
                - results["female_metrics"]["mae"],
                "r2_difference": results["male_metrics"]["r2_score"]
                - results["female_metrics"]["r2_score"],
                "correlation_difference": results["male_metrics"]["pearson_correlation"]
                - results["female_metrics"]["pearson_correlation"],
            }

        return results

    def evaluate_trajectory_quality(self, trajectory_results: dict) -> dict[str, float]:
        """
        Evaluate quality of aging trajectory analysis.

        Args:
            trajectory_results: Results from trajectory analysis

        Returns:
            Dictionary with trajectory quality metrics
        """
        metrics = {}

        all_trajectories = trajectory_results["all_trajectories"]
        significant_trajectories = trajectory_results["significant_trajectories"]
        method = trajectory_results["method"]

        # Basic statistics
        metrics["total_genes_analyzed"] = len(all_trajectories)
        metrics["significant_genes_count"] = len(significant_trajectories)
        metrics["significant_gene_percentage"] = (
            len(significant_trajectories) / len(all_trajectories)
        ) * 100

        if method == "linear":
            # Linear trajectory metrics
            correlations = [t["correlation"] for t in all_trajectories.values()]
            [t["p_value"] for t in all_trajectories.values()]

            metrics["mean_absolute_correlation"] = np.mean(np.abs(correlations))
            metrics["median_absolute_correlation"] = np.median(np.abs(correlations))
            metrics["max_correlation"] = np.max(np.abs(correlations))

            # Count trajectory directions
            increasing_count = sum(
                1 for t in significant_trajectories.values() if t["correlation"] > 0
            )
            decreasing_count = len(significant_trajectories) - increasing_count

            metrics["increasing_trajectories"] = increasing_count
            metrics["decreasing_trajectories"] = decreasing_count
            metrics["increasing_percentage"] = (
                (increasing_count / len(significant_trajectories)) * 100
                if len(significant_trajectories) > 0
                else 0
            )

        elif method == "polynomial":
            # Polynomial trajectory metrics
            r_squared_values = [t["r_squared"] for t in all_trajectories.values()]

            metrics["mean_r_squared"] = np.mean(r_squared_values)
            metrics["median_r_squared"] = np.median(r_squared_values)
            metrics["max_r_squared"] = np.max(r_squared_values)

            # Count trajectory types
            trajectory_types = [
                t["trajectory_type"] for t in significant_trajectories.values()
            ]
            type_counts = {}
            for ttype in [
                "linear_increasing",
                "linear_decreasing",
                "u_shaped",
                "inverted_u",
            ]:
                count = trajectory_types.count(ttype)
                type_counts[f"{ttype}_count"] = count
                type_counts[f"{ttype}_percentage"] = (
                    (count / len(trajectory_types)) * 100
                    if len(trajectory_types) > 0
                    else 0
                )

            metrics.update(type_counts)

        return metrics

    def evaluate_aging_marker_performance(
        self, adata, aging_markers: list[str], age_column: str = "age"
    ) -> dict[str, Any]:
        """
        Evaluate how well known aging markers behave in the data.

        Args:
            adata: AnnData object
            aging_markers: List of known aging marker genes
            age_column: Column containing age information

        Returns:
            Dictionary with aging marker evaluation metrics
        """
        results = {}

        available_markers = [gene for gene in aging_markers if gene in adata.var.index]
        results["available_markers"] = available_markers
        results["missing_markers"] = [
            gene for gene in aging_markers if gene not in adata.var.index
        ]
        results["marker_coverage"] = len(available_markers) / len(aging_markers)

        # Analyze each available marker
        marker_analysis = {}
        ages = adata.obs[age_column].values

        for marker in available_markers:
            marker_idx = list(adata.var.index).index(marker)
            expression = (
                adata.X[:, marker_idx].toarray().flatten()
                if hasattr(adata.X, "toarray")
                else adata.X[:, marker_idx]
            )

            # Correlation with age
            correlation, p_value = stats.pearsonr(ages, expression)

            # Expression variability across ages
            age_groups = {}
            for age in adata.obs[age_column].unique():
                age_mask = adata.obs[age_column] == age
                age_expression = expression[age_mask]
                age_groups[str(age)] = {
                    "mean": float(np.mean(age_expression)),
                    "std": float(np.std(age_expression)),
                    "cv": float(np.std(age_expression) / np.mean(age_expression))
                    if np.mean(age_expression) > 0
                    else 0,
                }

            marker_analysis[marker] = {
                "age_correlation": correlation,
                "correlation_p_value": p_value,
                "age_group_expression": age_groups,
                "overall_expression_mean": float(np.mean(expression)),
                "overall_expression_std": float(np.std(expression)),
            }

        results["marker_analysis"] = marker_analysis

        # Summary statistics
        if marker_analysis:
            correlations = [m["age_correlation"] for m in marker_analysis.values()]
            results["summary"] = {
                "mean_age_correlation": np.mean(np.abs(correlations)),
                "strong_age_correlation_count": sum(
                    1 for c in correlations if abs(c) > 0.5
                ),
                "significant_correlation_count": sum(
                    1
                    for m in marker_analysis.values()
                    if m["correlation_p_value"] < 0.05
                ),
            }

        return results

    def _evaluate_age_group_consistency(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Evaluate consistency within age groups.

        Args:
            y_true: True age values
            y_pred: Predicted age values

        Returns:
            Dictionary with age group consistency metrics
        """
        metrics = {}

        # Define age groups
        age_groups = {"young": (0, 15), "middle": (15, 45), "old": (45, 100)}

        for group_name, (min_age, max_age) in age_groups.items():
            group_mask = (y_true >= min_age) & (y_true < max_age)

            if np.sum(group_mask) > 0:
                group_true = y_true[group_mask]
                group_pred = y_pred[group_mask]

                group_mae = mean_absolute_error(group_true, group_pred)
                group_r2 = r2_score(group_true, group_pred)

                metrics[f"{group_name}_mae"] = group_mae
                metrics[f"{group_name}_r2"] = group_r2
                metrics[f"{group_name}_sample_count"] = int(np.sum(group_mask))

        return metrics

    def calculate_aging_score(
        self, adata, aging_markers: list[str], age_column: str = "age"
    ) -> np.ndarray:
        """
        Calculate a composite aging score based on marker expression.

        Args:
            adata: AnnData object
            aging_markers: List of aging marker genes
            age_column: Column containing age information

        Returns:
            Array of aging scores for each cell
        """
        available_markers = [gene for gene in aging_markers if gene in adata.var.index]

        if not available_markers:
            return np.zeros(adata.n_obs)

        # Extract marker expressions
        marker_expressions = []
        for marker in available_markers:
            marker_idx = list(adata.var.index).index(marker)
            expression = (
                adata.X[:, marker_idx].toarray().flatten()
                if hasattr(adata.X, "toarray")
                else adata.X[:, marker_idx]
            )

            # Normalize expression (z-score)
            normalized_expr = (expression - np.mean(expression)) / np.std(expression)
            marker_expressions.append(normalized_expr)

        # Calculate composite score (mean of normalized expressions)
        aging_scores = np.mean(marker_expressions, axis=0)

        return aging_scores

    def evaluate_classification_and_regression(
        self, true_labels, predicted_classes, predictions
    ):
        """
        Comprehensive evaluation for classification tasks.

        Args:
            true_labels: True class labels
            predicted_classes: Predicted class labels
            predictions: Prediction probabilities (for AUC calculation)

        Returns:
            Dictionary containing all computed metrics
        """
        metrics = {}

        # Get metrics from config
        eval_config = getattr(self.config, "evaluation", {})
        config_metrics = eval_config.get("metrics", {})
        eval_metrics = config_metrics.get("evaluation", {}).get(
            "classification", ["accuracy", "f1_score", "precision", "recall", "auc"]
        )

        # Basic classification metrics
        if "accuracy" in eval_metrics:
            metrics["accuracy"] = float(accuracy_score(true_labels, predicted_classes))

        if "f1_score" in eval_metrics:
            # Use macro average for multiclass, binary for binary classification
            average = "macro" if len(np.unique(true_labels)) > 2 else "binary"
            metrics["f1_score"] = float(
                f1_score(true_labels, predicted_classes, average=average)
            )

        if "precision" in eval_metrics:
            average = "macro" if len(np.unique(true_labels)) > 2 else "binary"
            metrics["precision"] = float(
                precision_score(true_labels, predicted_classes, average=average)
            )

        if "recall" in eval_metrics:
            average = "macro" if len(np.unique(true_labels)) > 2 else "binary"
            metrics["recall"] = float(
                recall_score(true_labels, predicted_classes, average=average)
            )

        # AUC score (requires prediction probabilities)
        if "auc" in eval_metrics and len(predictions.shape) > 1:
            try:
                if predictions.shape[1] == 2:
                    # Binary classification - use probability of positive class
                    metrics["auc"] = float(
                        roc_auc_score(true_labels, predictions[:, 1])
                    )
                else:
                    # Multiclass classification - use macro average
                    metrics["auc"] = float(
                        roc_auc_score(
                            true_labels, predictions, multi_class="ovr", average="macro"
                        )
                    )
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics["auc"] = None

        # Confusion matrix
        if "confusion_matrix" in eval_metrics:
            cm = confusion_matrix(true_labels, predicted_classes)
            metrics["confusion_matrix"] = (
                cm.tolist()
            )  # Convert to list for JSON serialization

        # Additional detailed metrics
        try:
            # Classification report as dictionary
            class_report = classification_report(
                true_labels, predicted_classes, output_dict=True
            )
            metrics["classification_report"] = class_report
        except Exception as e:
            logger.warning(f"Could not compute classification report: {e}")

        # Class-wise metrics if multiclass
        unique_classes = np.unique(true_labels)
        if len(unique_classes) > 2:
            for class_id in unique_classes:
                class_mask = true_labels == class_id
                class_pred_mask = predicted_classes == class_id

                # Per-class precision, recall, f1
                if np.sum(class_pred_mask) > 0:  # Avoid division by zero
                    class_precision = np.sum(class_mask & class_pred_mask) / np.sum(
                        class_pred_mask
                    )
                    metrics[f"class_{class_id}_precision"] = float(class_precision)

                if np.sum(class_mask) > 0:  # Avoid division by zero
                    class_recall = np.sum(class_mask & class_pred_mask) / np.sum(
                        class_mask
                    )
                    metrics[f"class_{class_id}_recall"] = float(class_recall)

                    if class_precision + class_recall > 0:
                        class_f1 = (
                            2
                            * (class_precision * class_recall)
                            / (class_precision + class_recall)
                        )
                        metrics[f"class_{class_id}_f1"] = float(class_f1)

        # Add baseline comparison if enabled
        eval_config = getattr(self.config, "evaluation", {})
        config_baselines = eval_config.get("metrics", {}).get("baselines", {})
        if config_baselines.get("enabled", False):
            baseline_metrics = self._compute_baseline_comparison(
                true_labels, predicted_classes, predictions
            )
            metrics["baselines"] = baseline_metrics

        # Log key metrics in a clean single line
        key_metrics = ["accuracy", "f1_score", "precision", "recall", "auc"]
        metric_values = []
        for metric in key_metrics:
            if metric in metrics and metrics[metric] is not None:
                metric_values.append(f"{metric}={metrics[metric]:.3f}")

        if metric_values:
            print("\n" + "=" * 60)
            print("🎯 MODEL PERFORMANCE")
            print("=" * 60)

            # Create a dynamic results table that adjusts to content length
            result_content = " | ".join(metric_values)
            table_width = max(
                60, len(result_content) + 8
            )  # Minimum 60, or content + padding

            # Top border
            print("┌" + "─" * (table_width - 2) + "┐")

            # Header
            header = "🤖 MODEL RESULTS"
            header_padding_left = (table_width - 2 - len(header)) // 2
            header_padding_right = table_width - 2 - len(header) - header_padding_left
            print(f"│{' ' * header_padding_left}{header}{' ' * header_padding_right}│")

            # Middle border
            print("├" + "─" * (table_width - 2) + "┤")

            # Results content
            content_padding_left = (table_width - 2 - len(result_content)) // 2
            content_padding_right = (
                table_width - 2 - len(result_content) - content_padding_left
            )
            print(
                f"│{' ' * content_padding_left}{result_content}{' ' * content_padding_right}│"
            )

            # Bottom border
            print("└" + "─" * (table_width - 2) + "┘")

        return metrics

    def _compute_baseline_comparison(self, true_labels, predicted_classes, predictions):
        """
        Compute baseline comparisons for classification tasks.

        Args:
            true_labels: True class labels
            predicted_classes: Model's predicted class labels
            predictions: Model's prediction probabilities

        Returns:
            Dictionary containing baseline metrics and comparisons
        """
        baselines = {}

        # Get baseline types from config
        eval_config = getattr(self.config, "evaluation", {})
        config_baselines = eval_config.get("metrics", {}).get("baselines", {})
        baseline_types = config_baselines.get("classification", [])

        print("\n" + "=" * 60)
        print("📊 BASELINE COMPARISON")
        print("=" * 60)

        # Use actual evaluation holdout data for baseline comparison
        test_X = self.test_data

        # Print table header with dynamic width for baseline method names
        if not baseline_types:
            # No baselines configured, return early
            return {}

        # Filter out empty baseline types and calculate max length
        valid_baseline_types = [bt for bt in baseline_types if bt and bt.strip()]
        if not valid_baseline_types:
            # No valid baselines configured, return early
            return {}

        max_method_len = max(
            len(baseline_type.replace("_", " ").title())
            for baseline_type in valid_baseline_types
        )
        method_width = max(
            20, max_method_len + 2
        )  # At least 20, or method name + padding

        # Create the table format strings
        header_format = f"│ {{:<{method_width}}} │ {{:^5}} │ {{:^5}} │ {{:^9}} │ {{:^6}} │ {{:^8}} │ {{:^7}} │"
        border_top = f"┌{'─' * (method_width + 2)}┬───────┬───────┬───────────┬────────┬──────────┬─────────┐"
        border_mid = f"├{'─' * (method_width + 2)}┼───────┼───────┼───────────┼────────┼──────────┼─────────┤"
        border_bot = f"└{'─' * (method_width + 2)}┴───────┴───────┴───────────┴────────┴──────────┴─────────┘"

        print(border_top)
        print(
            header_format.format(
                "Baseline Method", "Acc", "F1", "Precision", "Recall", "Acc Δ", "F1 Δ"
            )
        )
        print(border_mid)

        for baseline_type in valid_baseline_types:
            try:
                if baseline_type == "random_classifier":
                    # Uniform random predictions
                    dummy_clf = DummyClassifier(strategy="uniform", random_state=42)
                    dummy_clf.fit(test_X, true_labels)
                    baseline_pred = dummy_clf.predict(test_X)

                elif baseline_type == "majority_class":
                    # Always predict the most frequent class
                    dummy_clf = DummyClassifier(strategy="most_frequent")
                    dummy_clf.fit(test_X, true_labels)
                    baseline_pred = dummy_clf.predict(test_X)

                elif baseline_type == "stratified_random":
                    # Random predictions respecting class distribution
                    dummy_clf = DummyClassifier(strategy="stratified", random_state=42)
                    dummy_clf.fit(test_X, true_labels)
                    baseline_pred = dummy_clf.predict(test_X)

                else:
                    logger.warning(f"Unknown baseline type: {baseline_type}")
                    continue

                # Compute metrics for this baseline
                baseline_accuracy = float(accuracy_score(true_labels, baseline_pred))
                baseline_f1 = float(
                    f1_score(
                        true_labels,
                        baseline_pred,
                        average="macro"
                        if len(np.unique(true_labels)) > 2
                        else "binary",
                    )
                )
                baseline_precision = float(
                    precision_score(
                        true_labels,
                        baseline_pred,
                        average="macro"
                        if len(np.unique(true_labels)) > 2
                        else "binary",
                    )
                )
                baseline_recall = float(
                    recall_score(
                        true_labels,
                        baseline_pred,
                        average="macro"
                        if len(np.unique(true_labels)) > 2
                        else "binary",
                    )
                )

                # Store baseline metrics
                baselines[baseline_type] = {
                    "accuracy": baseline_accuracy,
                    "f1_score": baseline_f1,
                    "precision": baseline_precision,
                    "recall": baseline_recall,
                }

                # Compute improvement over baseline
                model_accuracy = float(accuracy_score(true_labels, predicted_classes))
                model_f1 = float(
                    f1_score(
                        true_labels,
                        predicted_classes,
                        average="macro"
                        if len(np.unique(true_labels)) > 2
                        else "binary",
                    )
                )

                baselines[baseline_type]["improvement"] = {
                    "accuracy_improvement": model_accuracy - baseline_accuracy,
                    "f1_improvement": model_f1 - baseline_f1,
                    "accuracy_ratio": model_accuracy / baseline_accuracy
                    if baseline_accuracy > 0
                    else float("inf"),
                    "f1_ratio": model_f1 / baseline_f1
                    if baseline_f1 > 0
                    else float("inf"),
                }

                baseline_name = baseline_type.replace("_", " ").title()
                acc_improvement = model_accuracy - baseline_accuracy
                f1_improvement = model_f1 - baseline_f1

                # Format improvements with + or - signs
                acc_sign = "+" if acc_improvement >= 0 else ""
                f1_sign = "+" if f1_improvement >= 0 else ""

                # Use the dynamic format string for consistent alignment
                row_format = f"│ {{:<{method_width}}} │ {{:.3f}} │ {{:.3f}} │ {{:>9.3f}} │ {{:.3f}} │ {{:>8s}} │ {{:>7s}} │"
                print(
                    row_format.format(
                        baseline_name,
                        baseline_accuracy,
                        baseline_f1,
                        baseline_precision,
                        baseline_recall,
                        f"{acc_sign}{acc_improvement:+.3f}",
                        f"{f1_sign}{f1_improvement:+.3f}",
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to compute {baseline_type} baseline: {e}")

        # Close the table with dynamic border
        print(border_bot)

        return baselines
