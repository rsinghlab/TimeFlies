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
        pipeline_mode="training",
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
            pipeline_mode: "training" (train+eval) or "evaluation" (eval-only)
        """
        self.config = config
        self.model = model
        self.test_data = test_data
        self.test_labels = test_labels
        self.label_encoder = label_encoder
        self.path_manager = path_manager
        self.result_type = result_type
        self.output_dir = output_dir
        self.pipeline_mode = pipeline_mode
        
        # Store training classes if available for classification type determination
        self.training_classes = None
        if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
            self.training_classes = self.label_encoder.classes_

    def compute_metrics(self):
        """
        Compute and save metrics using the model and test data.

        This method is called by the pipeline to evaluate model performance.
        """
        if self.model is None or self.test_data is None or self.test_labels is None:
            logger.warning("Missing model or test data, skipping metrics computation")
            return

        # Display evaluation dataset information if in eval-only pipeline
        if self.pipeline_mode == "evaluation":
            self._display_evaluation_info()

        # Make predictions
        predictions = self.model.predict(self.test_data, verbose=0)
        
        # Get task type from config
        task_type = getattr(self.config.model, "task_type", "classification")

        # Process predictions based on task type
        if task_type == "regression":
            # For regression, predictions are already continuous values
            predicted_classes = predictions.flatten()
        else:
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

        # Process true labels based on task type
        if task_type == "regression":
            # For regression, labels are already continuous - just flatten
            if len(true_labels.shape) > 1:
                true_labels = true_labels.flatten()
        else:
            # For classification - if true_labels is 2D (one-hot encoded), get the class indices
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

        # Compute metrics based on task type from config
        task_type = getattr(self.config.model, "task_type", "classification")
        if task_type == "regression":
            # Pure regression task - treat age as continuous
            metrics = self.evaluate_age_prediction(true_labels, predictions.flatten())
        else:
            # Classification task (default) - compute both classification and regression metrics
            # This allows for age classification while also tracking regression-style metrics
            metrics = self.evaluate_classification_and_regression(
                true_labels, predicted_classes, predictions
            )

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

            # Add genotype information - use the processed test data directly
            try:
                # First check if test_data is an AnnData object with genotype info
                if (hasattr(self.test_data, 'obs') and 
                    hasattr(self.test_data.obs, 'columns') and 
                    "genotype" in self.test_data.obs.columns):
                    
                    # Use genotypes directly from the processed test data that was used for prediction
                    individual_genotypes = self.test_data.obs["genotype"].tolist()
                    
                    if len(individual_genotypes) == len(predictions_df):
                        predictions_df["genotype"] = individual_genotypes
                    else:
                        predictions_df["genotype"] = "unknown"
                        
                elif hasattr(self, 'test_data_genotypes'):
                    # Use stored genotype info if available
                    predictions_df["genotype"] = self.test_data_genotypes
                else:
                    # Fallback: try to get from the processed evaluation data that was used
                    from common.data.loaders import DataLoader
                    data_loader = DataLoader(self.config)
                    _, adata_eval, _ = data_loader.load_data()
                    
                    if adata_eval is not None and "genotype" in adata_eval.obs.columns:
                        split_config = getattr(self.config.data, "split", None)
                        if split_config and hasattr(split_config, "test"):
                            test_values = getattr(split_config, "test", [])
                            # Handle case mismatch
                            test_values_lower = [str(v).lower() for v in test_values]
                            
                            # Create case-insensitive mask
                            genotype_lower = adata_eval.obs["genotype"].str.lower()
                            mask = genotype_lower.isin(test_values_lower)
                            
                            if mask.sum() > 0:
                                # Get the original case genotypes that match
                                individual_genotypes = adata_eval.obs.loc[mask, "genotype"].tolist()
                                
                                if len(individual_genotypes) == len(predictions_df):
                                    predictions_df["genotype"] = individual_genotypes
                                else:
                                    predictions_df["genotype"] = "unknown"
                            else:
                                predictions_df["genotype"] = "unknown"
                        else:
                            predictions_df["genotype"] = "unknown"
                    else:
                        predictions_df["genotype"] = "unknown"
                        
            except Exception as e:
                print(f"Error extracting genotypes: {e}")
                predictions_df["genotype"] = "unknown"

            # Save to CSV
            predictions_file = os.path.join(results_dir, "predictions.csv")
            predictions_df.to_csv(predictions_file, index=False)
            # Predictions saved

        # Display baseline comparisons if they were computed
        if "baselines" in metrics and metrics["baselines"]:
            pass  # The display was already done in _compute_baseline_comparison

        return metrics

    def _determine_classification_type(self, true_labels, predicted_classes):
        """
        Determine whether to use binary or multiclass metrics.
        
        Args:
            true_labels: Actual labels from test data
            predicted_classes: Model predictions
            
        Returns:
            "binary" or "multiclass"
        """
        # Get config setting
        eval_config = getattr(self.config, "evaluation", {})
        metrics_config = eval_config.get("metrics", {})
        classification_type_config = metrics_config.get("classification_type", "auto")
        
        # Handle forced modes
        if classification_type_config == "binary":
            return "binary"
        elif classification_type_config == "multiclass":
            return "multiclass"
        
        # Auto detection logic
        # Get unique classes in test data
        test_classes = set(np.unique(true_labels))
        num_test_classes = len(test_classes)
        
        # Get training classes (what model learned)
        if self.training_classes is not None:
            train_classes = set(self.training_classes)
            num_train_classes = len(train_classes)
        else:
            # If no label encoder, infer from predictions
            all_predicted = set(np.unique(predicted_classes))
            train_classes = test_classes.union(all_predicted)
            num_train_classes = len(train_classes)
        
        # Binary only if:
        # 1. Model was trained on exactly 2 classes
        # 2. Test data has exactly 2 classes  
        # 3. Test classes are exactly the same as training classes
        if num_train_classes == 2 and num_test_classes == 2 and test_classes == train_classes:
            return "binary"
        else:
            return "multiclass"
    
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
            if adata_eval is None or "genotype" not in adata_eval.obs.columns:
                return None

            # Apply same filtering that was used during evaluation to match sample order
            split_config = getattr(self.config.data, "split", None)
            if split_config and hasattr(split_config, "test"):
                test_genotypes = getattr(split_config, "test", [])
                if test_genotypes:
                    # Filter to only test genotypes and return individual genotypes
                    mask = adata_eval.obs["genotype"].isin(test_genotypes)
                    # Return the actual individual genotypes for each sample
                    filtered_genotypes = adata_eval.obs.loc[mask, "genotype"].tolist()
                    return filtered_genotypes

            # Fallback: return all genotypes (each sample gets its own genotype)
            return adata_eval.obs["genotype"].tolist()

        except Exception as e:
            # Debug: print the error for troubleshooting
            print(f"DEBUG: Could not load genotype info: {e}")
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
        Evaluate prediction consistency for each actual age.

        Args:
            y_true: True age values
            y_pred: Predicted age values

        Returns:
            Dictionary with per-age consistency metrics
        """
        metrics = {}

        # Analyze performance for each actual age
        unique_ages = np.unique(y_true)

        for age in unique_ages:
            age_mask = y_true == age
            if np.sum(age_mask) > 0:
                age_true = y_true[age_mask]
                age_pred = y_pred[age_mask]

                age_mae = mean_absolute_error(age_true, age_pred)

                # Only compute R² if there's variation in predictions
                if len(np.unique(age_pred)) > 1:
                    age_r2 = r2_score(age_true, age_pred)
                else:
                    age_r2 = 0.0  # No variation in predictions

                age_key = f"age_{int(age)}d"  # e.g., "age_10d", "age_20d"
                metrics[f"{age_key}_mae"] = age_mae
                metrics[f"{age_key}_r2"] = age_r2
                metrics[f"{age_key}_sample_count"] = int(np.sum(age_mask))

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
        # Determine classification type based on config
        classification_type = self._determine_classification_type(true_labels, predicted_classes)
        # Basic classification metrics
        if "accuracy" in eval_metrics:
            metrics["accuracy"] = float(accuracy_score(true_labels, predicted_classes))

        if "f1_score" in eval_metrics:
            if classification_type == "binary":
                # Binary classification - use binary average
                unique_classes = np.unique(true_labels)
                pos_label = unique_classes[-1]  # Use higher age as positive
                metrics["f1_score"] = float(
                    f1_score(true_labels, predicted_classes, average="binary", pos_label=pos_label)
                )
            else:
                # Multiclass - use macro average
    
                metrics["f1_score"] = float(
                    f1_score(true_labels, predicted_classes, average="macro")
                )


        if "precision" in eval_metrics:

            if classification_type == "binary":
                unique_classes = np.unique(true_labels)
                pos_label = unique_classes[-1]
                metrics["precision"] = float(
                    precision_score(true_labels, predicted_classes, average="binary", pos_label=pos_label)
                )
            else:
                metrics["precision"] = float(
                    precision_score(true_labels, predicted_classes, average="macro")
                )


        if "recall" in eval_metrics:

            if classification_type == "binary":
                unique_classes = np.unique(true_labels)
                pos_label = unique_classes[-1]
                metrics["recall"] = float(
                    recall_score(true_labels, predicted_classes, average="binary", pos_label=pos_label)
                )
            else:
                metrics["recall"] = float(
                    recall_score(true_labels, predicted_classes, average="macro")
                )


        # AUC score (requires prediction probabilities)
        if "auc" in eval_metrics and len(predictions.shape) > 1:
            try:
                # Multiclass AUC with proper class handling
                
                # Convert true labels to match the training classes for AUC calculation
                if self.label_encoder and hasattr(self.label_encoder, 'classes_'):
                    # Use the label encoder to transform the eval labels to indices
                    # This ensures consistency with training
                    try:
                        # Convert float labels to strings to match label_encoder format
                        string_labels = [str(int(label)) if isinstance(label, float) else str(label) for label in true_labels]
                        true_indices = self.label_encoder.transform(string_labels)
                        
                        metrics["auc"] = float(
                            roc_auc_score(true_indices, predictions, multi_class="ovr", average="macro")
                        )
                    except ValueError as e:
                        # If label encoder fails (unknown labels), add specific AUC not
                        # Fallback to binary AUC using just the first two classes
                        if predictions.shape[1] >= 2:
                            metrics["auc"] = float(roc_auc_score(true_labels, predictions[:, 1]))
                        else:
                            metrics["auc"] = 0.0
                else:
                    # Fallback to binary AUC if no label encoder
                    if predictions.shape[1] >= 2:
                        metrics["auc"] = float(
                            roc_auc_score(true_labels, predictions[:, 1], multi_class="raise")
                        )
                    else:
                        metrics["auc"] = 0.0

                # AUC calculation complete
                    
                    # Use only the relevant prediction columns for AUC calculation
                    relevant_predictions = predictions[:, eval_indices]
                    
                    if num_eval_classes == 2:
                        # Binary AUC - use positive class probability
                        pos_idx = 1 if len(eval_indices) == 2 else 0
                        metrics["auc"] = float(
                            roc_auc_score(true_labels, relevant_predictions[:, pos_idx])
                        )
                    else:
                        # Multiclass AUC
                        metrics["auc"] = float(
                            roc_auc_score(
                                true_labels, relevant_predictions, 
                                multi_class="ovr", average="macro"
                            )
                        )
            except Exception as e:
                logger.warning(f"Could not compute AUC: {e}")
                metrics["auc"] = 0.0


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

        # Display model performance first in a clean, simple format
        key_metrics = ["accuracy", "f1_score", "precision", "recall", "auc"]
        metric_values = []
        for metric in key_metrics:
            if metric in metrics and metrics[metric] is not None:
                capitalized_metric = metric.replace("_", " ").title()
                metric_values.append(f"{capitalized_metric}: {metrics[metric]:.3f}")


        # Add baseline comparison if enabled (after model performance)
        eval_config = getattr(self.config, "evaluation", {})
        config_baselines = eval_config.get("metrics", {}).get("baselines", {})

        if config_baselines.get("enabled", False):
            baseline_metrics = self._compute_baseline_comparison(
                true_labels, predicted_classes, predictions
            )
            metrics["baselines"] = baseline_metrics

    
        return metrics

    def _display_evaluation_info(self):
        """Display comprehensive information about the evaluation dataset."""
        print("\n" + "=" * 80)
        print("📊 EVALUATION DATASET ANALYSIS")
        print("=" * 80)

        # Dataset dimensions - handle different shapes safely
        if len(self.test_data.shape) == 2:
            n_samples, n_features = self.test_data.shape
        elif len(self.test_data.shape) == 1:
            n_samples = self.test_data.shape[0]
            n_features = 1
        else:
            n_samples = self.test_data.shape[0]
            n_features = np.prod(self.test_data.shape[1:])

        # Basic dataset info
        print("Dataset Size:")
        print(f"  └─ Samples:           {n_samples:,}")
        print(f"  └─ Features (genes):  {n_features:,}")

        # Class distribution analysis
        # Convert one-hot encoded labels to class indices if necessary
        if len(self.test_labels.shape) > 1 and self.test_labels.shape[1] > 1:
            # One-hot encoded - convert to class indices
            class_labels = np.argmax(self.test_labels, axis=1)
        else:
            # Already class indices - but if using label encoder, convert to encoded indices
            class_labels = self.test_labels
            if self.label_encoder and hasattr(self.label_encoder, "classes_"):
                # Convert original labels to encoded indices for display
                try:
                    # Check if class_labels contains original values that need encoding
                    original_values = np.unique(class_labels)
                    encoded_values = np.unique(
                        self.label_encoder.transform(class_labels)
                    )
                    if not np.array_equal(original_values, encoded_values):
                        class_labels = self.label_encoder.transform(class_labels)
                except (ValueError, AttributeError):
                    # If transformation fails, keep original labels
                    pass

        unique_labels, counts = np.unique(class_labels, return_counts=True)
        total_label_count = (
            counts.sum()
        )  # Use actual label count for correct percentages

        print("\nClass Distribution:")

        # Get class names if label encoder is available
        if self.label_encoder and hasattr(self.label_encoder, "classes_"):
            class_names = self.label_encoder.classes_
            for i, (label, count) in enumerate(zip(unique_labels, counts)):
                if i < len(class_names):
                    class_name = class_names[int(label)]
                    percentage = (count / total_label_count) * 100
                    print(
                        f"  └─ {class_name:<12}: {count:>5,} samples ({percentage:>5.1f}%)"
                    )
                else:
                    percentage = (count / total_label_count) * 100
                    print(
                        f"  └─ Class {label:<7}: {count:>5,} samples ({percentage:>5.1f}%)"
                    )
        else:
            for label, count in zip(unique_labels, counts):
                percentage = (count / total_label_count) * 100
                print(f"  └─ {label:<12}: {count:>5,} samples ({percentage:>5.1f}%)")

        # Add genotype/split information if available from config
        self._display_split_info()

        print("=" * 80)

    def _display_split_info(self):
        """Display split configuration and genotype information if available."""
        try:
            # Check if split configuration is available
            if hasattr(self.config, "data") and hasattr(self.config.data, "split"):
                split_config = self.config.data.split

                print("\nSplit Configuration:")

                # Split method
                if hasattr(split_config, "method"):
                    method = split_config.method
                    print(f"  └─ Split Method:      {method}")

                    if method == "column":
                        # Column-based splitting details
                        if hasattr(split_config, "column"):
                            split_column = split_config.column
                            print(f"  └─ Split Column:      {split_column}")

                        if hasattr(split_config, "train"):
                            train_values = split_config.train
                            if isinstance(train_values, list):
                                train_str = ", ".join(map(str, train_values))
                            else:
                                train_str = str(train_values)
                            print(f"  └─ Training Values:   {train_str}")

                        if hasattr(split_config, "test"):
                            test_values = split_config.test
                            if isinstance(test_values, list):
                                test_str = ", ".join(map(str, test_values))
                            else:
                                test_str = str(test_values)
                            print(f"  └─ Test Values:       {test_str}")

                    elif method == "random":
                        # Random splitting details
                        if hasattr(split_config, "test_ratio"):
                            test_ratio = split_config.test_ratio
                            train_ratio = 1.0 - test_ratio
                            print(f"  └─ Training Split:    {train_ratio:.1%}")
                            print(f"  └─ Test Split:        {test_ratio:.1%}")

                    # If we have genotype info in metadata, show distribution
                    if hasattr(self.config, "metadata") and hasattr(
                        self.config.metadata, "genotype_distribution"
                    ):
                        print("\nGenotype Distribution (from metadata):")
                        genotype_dist = self.config.metadata.genotype_distribution
                        for genotype, count in genotype_dist.items():
                            print(f"  └─ {genotype:<12}: {count:>5,} samples")

            # Additional sample information
            if hasattr(self.config, "data") and hasattr(self.config.data, "sampling"):
                sampling = self.config.data.sampling
                if hasattr(sampling, "sex") and sampling.sex != "both":
                    print("\nSample Filtering:")
                    print(f"  └─ Sex Filter:        {sampling.sex}")
                if hasattr(sampling, "age_range"):
                    print(f"  └─ Age Range:         {sampling.age_range}")

        except Exception:
            # If config parsing fails, silently skip additional info
            pass

    def _compute_baseline_comparison(self, true_labels, predicted_classes, predictions):
        """
        Compute baseline comparisons for classification or regression tasks.

        Args:
            true_labels: True labels/values
            predicted_classes: Model's predicted labels/values
            predictions: Model's prediction probabilities (for classification) or raw predictions (for regression)

        Returns:
            Dictionary containing baseline metrics and comparisons
        """
        baselines = {}

        # Get task type and baseline types from config
        task_type = getattr(self.config.model, "task_type", "classification")
        eval_config = getattr(self.config, "evaluation", {})
        config_baselines = eval_config.get("metrics", {}).get("baselines", {})
        
        if task_type == "regression":
            baseline_types = config_baselines.get("regression", ["mean_baseline", "median_baseline"])
        else:
            baseline_types = config_baselines.get("classification", [])

        print("\n MODEL COMPARISON (Holdout Evaluation)")
        print("=" * 80)
        
        
        # Add warning if evaluation classes don't match training classes
        if hasattr(self, 'label_encoder') and self.label_encoder and hasattr(self.label_encoder, 'classes_'):
            training_classes = set(self.label_encoder.classes_)
            eval_classes = set(str(int(label)) if isinstance(label, float) else str(label) for label in np.unique(true_labels))
            if training_classes != eval_classes:
                missing_in_eval = training_classes - eval_classes
                if missing_in_eval:
                    print("⚠️  WARNING: Model trained on {} classes but evaluation data only contains {} classes.".format(len(training_classes), len(eval_classes)))
                    print("   Missing from evaluation: {}".format(', '.join(sorted(missing_in_eval))))
                    print("   Metrics may underestimate model performance - predictions of missing classes are marked as 'wrong'.")
                    print("   AUC calculation uses fallback method due to class mismatch.")
                    print("   For accelerated aging analysis, examine the predictions.csv file instead.")
                    print()

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

        # Get number of classes for classification tasks
        num_classes = len(np.unique(true_labels))
        
        # Get model performance based on task type
        if task_type == "regression":
            # Regression metrics
            model_mae = float(mean_absolute_error(true_labels, predicted_classes))
            model_rmse = float(np.sqrt(mean_squared_error(true_labels, predicted_classes)))
            model_r2 = float(r2_score(true_labels, predicted_classes))
        else:
            # Classification metrics
            model_accuracy = float(accuracy_score(true_labels, predicted_classes))
            
            # Determine classification type for baseline metrics
            classification_type = self._determine_classification_type(true_labels, predicted_classes)
            
            if classification_type == "binary":
                unique_classes = np.unique(true_labels)
                pos_label = unique_classes[-1]
                model_f1 = float(f1_score(true_labels, predicted_classes, average="binary", pos_label=pos_label))
                model_precision = float(precision_score(true_labels, predicted_classes, average="binary", pos_label=pos_label))
                model_recall = float(recall_score(true_labels, predicted_classes, average="binary", pos_label=pos_label))
            else:
                model_f1 = float(f1_score(true_labels, predicted_classes, average="macro"))
                model_precision = float(precision_score(true_labels, predicted_classes, average="macro"))
                model_recall = float(recall_score(true_labels, predicted_classes, average="macro"))

        # Calculate AUC for our model
        try:
            if num_classes == 2 and len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Binary classification with multiple prediction columns
                model_auc = float(roc_auc_score(true_labels, predictions[:, 1]))
            elif num_classes == 2:
                # Binary classification - use flattened predictions or single column
                pred_values = predictions.flatten() if len(predictions.shape) > 1 else predictions
                model_auc = float(roc_auc_score(true_labels, pred_values))
            else:
                # Multiclass classification
                if len(predictions.shape) > 1:
                    model_auc = float(
                        roc_auc_score(true_labels, predictions, multi_class="ovr")
                    )
                else:
                    model_auc = 0.0
        except Exception as e:
            model_auc = 0.0

        max_method_len = max(
            len(baseline_type.replace("_", " ").title())
            for baseline_type in valid_baseline_types + ["**Our Model**"]
        )
        method_width = max(
            20, max_method_len + 2
        )  # At least 20, or method name + padding

        # Create the table format strings - ensure header and border widths match
        # Border widths: 7, 7, 11, 8, 8, 10, 9 (including padding)
        header_format = f"│ {{:<{method_width}}} │ {{:^5}} │ {{:^5}} │ {{:^9}} │ {{:^6}} │ {{:^6}} │ {{:^8}} │ {{:^7}} │"
        row_format = f"│ {{:<{method_width}}} │ {{:^5}} │ {{:^5}} │ {{:^9}} │ {{:^6}} │ {{:^6}} │ {{:^8}} │ {{:^7}} │"
        border_top = f"┌{'─' * (method_width + 2)}┬───────┬───────┬───────────┬────────┬────────┬──────────┬─────────┐"
        border_mid = f"├{'─' * (method_width + 2)}┼───────┼───────┼───────────┼────────┼────────┼──────────┼─────────┤"
        border_bot = f"└{'─' * (method_width + 2)}┴───────┴───────┴───────────┴────────┴────────┴──────────┴─────────┘"

        print(border_top)
        print(
            header_format.format(
                "Method", "Acc", "F1", "Precision", "Recall", "AUC", "Δ Acc", "Δ F1"
            )
        )
        print(border_mid)

        # Print our model row first
        print(
            row_format.format(
                "**Our Model**",
                f"{model_accuracy:.3f}",
                f"{model_f1:.3f}",
                f"{model_precision:.3f}",
                f"{model_recall:.3f}",
                f"{model_auc:.3f}",
                "–",
                "–",
            )
        )

        for baseline_type in valid_baseline_types:
            try:
                if task_type == "regression":
                    # Regression baselines
                    if baseline_type == "mean_baseline":
                        # Always predict mean
                        dummy_reg = DummyRegressor(strategy="mean")
                        dummy_reg.fit(test_X, true_labels)
                        baseline_pred = dummy_reg.predict(test_X)
                    
                    elif baseline_type == "median_baseline":
                        # Always predict median
                        dummy_reg = DummyRegressor(strategy="median")
                        dummy_reg.fit(test_X, true_labels)
                        baseline_pred = dummy_reg.predict(test_X)
                    
                    else:
                        logger.warning(f"Unknown regression baseline type: {baseline_type}")
                        continue
                        
                else:
                    # Classification baselines
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
                        logger.warning(f"Unknown classification baseline type: {baseline_type}")
                        continue

                # Compute metrics for this baseline based on task type
                if task_type == "regression":
                    # Regression metrics
                    baseline_mae = float(mean_absolute_error(true_labels, baseline_pred))
                    baseline_rmse = float(np.sqrt(mean_squared_error(true_labels, baseline_pred)))
                    baseline_r2 = float(r2_score(true_labels, baseline_pred))
                    
                    # Store baseline metrics
                    baselines[baseline_type] = {
                        "mae": baseline_mae,
                        "rmse": baseline_rmse,
                        "r2": baseline_r2,
                    }
                    
                    # Compute improvement over baseline
                    baselines[baseline_type]["improvement"] = {
                        "mae_improvement": baseline_mae - model_mae,  # Lower is better
                        "rmse_improvement": baseline_rmse - model_rmse,  # Lower is better
                        "r2_improvement": model_r2 - baseline_r2,  # Higher is better
                    }
                    
                else:
                    # Classification metrics
                    baseline_accuracy = float(accuracy_score(true_labels, baseline_pred))

                    # Use same classification type as main model evaluation
                    if classification_type == "binary":
                        unique_classes = np.unique(true_labels)
                        pos_label = unique_classes[-1]  # Higher age as positive
                        baseline_f1 = float(
                            f1_score(true_labels, baseline_pred, average="binary", pos_label=pos_label)
                        )
                        baseline_precision = float(
                            precision_score(true_labels, baseline_pred, average="binary", pos_label=pos_label)
                        )
                        baseline_recall = float(
                            recall_score(true_labels, baseline_pred, average="binary", pos_label=pos_label)
                        )
                    else:
                        # Multiclass - use macro average
                        baseline_f1 = float(
                            f1_score(true_labels, baseline_pred, average="macro")
                        )
                        baseline_precision = float(
                            precision_score(true_labels, baseline_pred, average="macro")
                        )
                        baseline_recall = float(
                            recall_score(true_labels, baseline_pred, average="macro")
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

                # Always use macro average for age prediction (multiclass problem)
                model_f1 = float(
                        f1_score(true_labels, predicted_classes, average="macro")
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
                acc_delta = f"{acc_improvement:+.3f}"
                f1_delta = f"{f1_improvement:+.3f}"

                # Calculate baseline AUC - use 0.5 for class mismatch scenarios
                # Check if there's a class mismatch (training vs eval classes)
                training_classes = set(self.label_encoder.classes_) if self.label_encoder and hasattr(self.label_encoder, 'classes_') else set()
                eval_classes = set(str(int(label)) if isinstance(label, float) else str(label) for label in np.unique(true_labels))
                
                if training_classes and training_classes != eval_classes:
                    # Class mismatch scenario - use theoretical random baseline
                    baseline_auc = 0.5
                else:
                    # No class mismatch - try to calculate actual baseline AUC
                    try:
                        if num_classes == 2:
                            # Binary classification
                            binary_true = np.array(true_labels)
                            binary_pred = np.array(baseline_pred)
                            if len(np.unique(binary_true)) == 2:
                                unique_vals = np.unique(binary_true)
                                binary_true = (binary_true == unique_vals[1]).astype(int)
                                binary_pred = (binary_pred == unique_vals[1]).astype(int)
                            baseline_auc = float(roc_auc_score(binary_true, binary_pred))
                        else:
                            # For multiclass without mismatch, still use 0.5 for simplicity
                            baseline_auc = 0.5
                    except Exception:
                        baseline_auc = 0.5
                print(
                    row_format.format(
                        baseline_name,
                        f"{baseline_accuracy:.3f}",
                        f"{baseline_f1:.3f}",
                        f"{baseline_precision:.3f}",
                        f"{baseline_recall:.3f}",
                        f"{baseline_auc:.3f}",
                        acc_delta,
                        f1_delta,
                    )
                )

            except Exception as e:
                logger.warning(f"Failed to compute {baseline_type} baseline: {e}")

        # Close the table with dynamic border
        print(border_bot)

        return baselines
