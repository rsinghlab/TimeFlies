#!/usr/bin/env python3
"""
TimeFlies Analysis Queue Runner
Runs analysis scripts on multiple model outputs.
"""

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisQueueRunner:
    """Runs analysis scripts on multiple models."""
    
    def __init__(self, project_dir: Path = None):
        """Initialize the analysis queue runner."""
        self.project_dir = project_dir or Path.cwd()
        self.results = []
        
        # Determine project-specific queue output directory with numbered subdirectories
        # Look for existing outputs structure to determine project name and path
        outputs_dir = self.project_dir / "outputs"
        if outputs_dir.exists():
            # Find the first project directory (e.g., fruitfly_alzheimers)
            project_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name != "analysis_summaries"]
            if project_dirs:
                project_name = project_dirs[0].name
                # Use the standard experiment structure: outputs/project/experiments/uncorrected/classification/queues/analysis/
                queues_base = outputs_dir / project_name / "experiments" / "uncorrected" / "classification" / "queues" / "analysis"
            else:
                # Fallback to generic location
                queues_base = outputs_dir / "queues" / "analysis"
        else:
            # Fallback if no outputs directory exists yet
            queues_base = self.project_dir / "outputs" / "queues" / "analysis"
            
        queues_base.mkdir(parents=True, exist_ok=True)
        
        # Find next available queue number
        existing_queues = [d for d in queues_base.iterdir() if d.is_dir() and d.name.startswith("queue_")]
        if existing_queues:
            queue_numbers = []
            for queue_dir in existing_queues:
                try:
                    num = int(queue_dir.name.split("_")[1])
                    queue_numbers.append(num)
                except (ValueError, IndexError):
                    continue
            next_queue_num = max(queue_numbers) + 1 if queue_numbers else 1
        else:
            next_queue_num = 1
            
        self.summary_dir = queues_base / f"queue_{next_queue_num}"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
    def _get_control_baseline_auc(self, model_dir: Path) -> float:
        """Find the corresponding ctrl-vs-ctrl model and get its AUC for quality assessment."""
        model_name = model_dir.name
        
        # Convert disease model name to control baseline name
        # Example: head_cnn_age_epithelial-cell_female_ctrl-vs-ab42 -> head_cnn_age_epithelial-cell_female_ctrl-vs-ctrl
        if "ctrl-vs-" in model_name and not model_name.endswith("ctrl-vs-ctrl"):
            # Replace the comparison part with ctrl-vs-ctrl
            parts = model_name.split("ctrl-vs-")
            if len(parts) == 2:
                control_name = parts[0] + "ctrl-vs-ctrl"
                
                # Find the control model directory
                control_dir = model_dir.parent / control_name
                if control_dir.exists():
                    # Load the control model's metrics
                    control_metrics_file = control_dir / "evaluation" / "metrics.json"
                    if control_metrics_file.exists():
                        try:
                            with open(control_metrics_file) as f:
                                control_metrics = json.load(f)
                            return control_metrics.get("auc", 0)
                        except Exception as e:
                            logger.warning(f"Could not load control metrics for {control_name}: {e}")
        
        # If this is already a ctrl-vs-ctrl model, return its own AUC
        if model_name.endswith("ctrl-vs-ctrl"):
            return 0  # Will use the model's own AUC from the calling function
        
        return 0  # No control baseline found
        
    def run_queue_with_models(self, model_list: List[str], analysis_script: str = None):
        """Run analysis queue on specific models from a list."""
        print("\nStarting Analysis Queue with Explicit Model List")
        print("="*60)
        
        # Find model directories for the specified models
        models_to_run = []
        base_dir = self.project_dir / "outputs"
        
        # Search for each model in the list
        for model_name in model_list:
            found = False
            for model_dir in self.find_model_outputs():
                if model_dir.name == model_name:
                    models_to_run.append(model_dir)
                    found = True
                    break
            
            if not found:
                print(f"Warning: Model '{model_name}' not found in outputs")
        
        print(f"Found {len(models_to_run)} models to analyze")
        print("")
        
        # Run analysis on each model
        for i, model_dir in enumerate(models_to_run, 1):
            print(f"[{i}/{len(models_to_run)}] Processing {model_dir.name}")
            print("")
            result = self.run_analysis_for_model(model_dir, Path(analysis_script) if analysis_script else None)
            self.results.append(result)
        
        # Generate summary
        print("")
        print("="*60)
        print("Generating Analysis Summary Report")
        print("="*60)
        
        self.generate_summary_report()
        self._print_summary()
        
    def _generate_full_name(self, metadata: Dict) -> str:
        """Generate a comprehensive experiment name from metadata."""
        parts = []
        
        # Architecture and target
        model_type = metadata.get("model_type", "Unknown")
        target = metadata.get("target", "Unknown")
        tissue = metadata.get("tissue", "Unknown")
        parts.append(f"{tissue}_{model_type}_{target}")
        
        # Cell type and sex
        data_filters = metadata.get("data_filters", {})
        cell_type = data_filters.get("cells", "Unknown").replace(" ", "-")
        sex = data_filters.get("sex", "Unknown")
        parts.append(f"{cell_type}_{sex}")
        
        # Comparison type
        split_config = metadata.get("split_config", {})
        comparison = split_config.get("split_name", "Unknown")
        parts.append(comparison)
        
        # Batch correction status
        if metadata.get("batch_correction", False):
            parts.append("batch-corrected")
        else:
            parts.append("uncorrected")
        
        return "_".join(parts)
        
    def run_analysis_for_model(self, model_dir: Path, analysis_script: Path) -> Dict:
        """Run analysis script for a single model."""
        result = {
            "model_name": model_dir.name,
            "model_path": str(model_dir),
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        
        # Load model metadata for enhanced reporting
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file) as f:
                    metadata = json.load(f)
                result["metadata"] = metadata
                
                # Extract key information for better reporting
                result["full_experiment_name"] = self._generate_full_name(metadata)
                result["model_architecture"] = metadata.get("model_type", "Unknown")
                result["target_variable"] = metadata.get("target", "Unknown")
                result["tissue_type"] = metadata.get("tissue", "Unknown")
                result["batch_corrected"] = metadata.get("batch_correction", False)
                
                # Data information
                data_filters = metadata.get("data_filters", {})
                result["sample_size"] = data_filters.get("samples", "Unknown")
                result["n_features"] = data_filters.get("variables", "Unknown")
                result["cell_type"] = data_filters.get("cells", "Unknown")
                result["sex"] = data_filters.get("sex", "Unknown")
                
                # Split configuration
                split_config = metadata.get("split_config", {})
                result["split_method"] = split_config.get("method", "Unknown")
                result["comparison"] = split_config.get("split_name", "Unknown")
                
                # Training details
                training = metadata.get("training", {})
                result["epochs_trained"] = training.get("epochs_run", "Unknown")
                result["best_val_loss"] = training.get("best_val_loss", "Unknown")
                
            except Exception as e:
                logger.warning(f"Could not load metadata for {model_dir.name}: {e}")
        
        # Load performance metrics if available
        metrics_file = model_dir / "evaluation" / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)
                result["performance_metrics"] = {
                    "accuracy": metrics.get("accuracy"),
                    "f1_score": metrics.get("f1_score"),
                    "auc": metrics.get("auc"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "mae": metrics.get("mae"),
                    "rmse": metrics.get("rmse"),
                    "r2_score": metrics.get("r2_score")
                }
                
                # Find corresponding ctrl-vs-ctrl model for quality assessment
                control_auc = self._get_control_baseline_auc(model_dir)
                
                # Use control model AUC for quality assessment (better measure of age prediction ability)
                auc_for_quality = control_auc if control_auc > 0 else metrics.get("auc", 0)
                
                # Determine model quality based on control baseline AUC
                if auc_for_quality > 0:
                    if auc_for_quality >= 0.85:
                        model_quality = "Excellent"
                        confidence = "High"
                    elif auc_for_quality >= 0.75:
                        model_quality = "Good"
                        confidence = "High"
                    elif auc_for_quality >= 0.65:
                        model_quality = "Fair"
                        confidence = "Medium"
                    elif auc_for_quality >= 0.55:
                        model_quality = "Poor"
                        confidence = "Low"
                    else:
                        model_quality = "Random"
                        confidence = "None"
                    result["model_quality"] = model_quality
                    result["prediction_confidence"] = confidence
                    result["control_auc"] = control_auc
                    result["disease_auc"] = metrics.get("auc", 0)
                    result["quality_metric"] = f"Control_AUC={auc_for_quality:.3f}"
                else:
                    result["model_quality"] = "Unknown"
                    result["prediction_confidence"] = "Unknown"
                    result["control_auc"] = 0
                    result["disease_auc"] = metrics.get("auc", 0)
                    result["quality_metric"] = "N/A"
            except Exception as e:
                logger.warning(f"Could not load performance metrics for {model_dir.name}: {e}")
        
        try:
            print(f"\n{'='*60}")
            print(f"Running analysis for: {model_dir.name}")
            print(f"{'='*60}")
            
            # Check if predictions exist
            predictions_file = model_dir / "evaluations" / "predictions.csv"
            if not predictions_file.exists():
                # Try alternative path
                predictions_file = model_dir / "evaluation" / "predictions.csv"
            if not predictions_file.exists():
                print(f"No predictions found for {model_dir.name}")
                result["status"] = "no_predictions"
                return result
            
            # Run the analysis script
            if analysis_script and analysis_script.exists():
                cmd = [sys.executable, str(analysis_script), str(model_dir)]
                print(f"Running: {' '.join(cmd)}")
                
                process_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    cwd=self.project_dir
                )
                
                if process_result.returncode == 0:
                    result["status"] = "completed"
                    result["analysis_output"] = process_result.stdout
                    print(f"Analysis completed for {model_dir.name}")
                else:
                    result["status"] = "failed"
                    result["error"] = process_result.stderr
                    print(f"Analysis failed for {model_dir.name}: {process_result.stderr}")
            else:
                result["status"] = "no_script"
                result["error"] = f"Analysis script not found: {analysis_script}"
                print(f"Analysis script not found: {analysis_script}")
                
        except Exception as e:
            logger.error(f"Analysis failed for {model_dir.name}: {e}")
            result["status"] = "failed"
            result["error"] = str(e)
            
        return result
    
    def find_model_outputs(self, pattern: str = "*") -> List[Path]:
        """Find all model output directories."""
        outputs_dir = self.project_dir / "outputs"
        if not outputs_dir.exists():
            return []
            
        # Look for directories with evaluation results recursively
        model_dirs = []
        
        # Use recursive glob to find all directories with predictions.csv
        for predictions_file in outputs_dir.rglob("predictions.csv"):
            if predictions_file.parent.name in ["evaluation", "evaluations"]:
                model_dir = predictions_file.parent.parent
                
                # Only include models from 'best' directory to avoid duplicates
                if "best" in model_dir.parts:
                    # Apply pattern filter to model directory name
                    if pattern == "*" or model_dir.match(pattern):
                        model_dirs.append(model_dir)
                
        return sorted(model_dirs)
    
    def run_queue(self, model_pattern: str = "*", analysis_script: str = None):
        """Run analysis queue on multiple models."""
        print("\nStarting Analysis Queue Runner")
        print("="*60)
        
        # Find all model outputs
        model_dirs = self.find_model_outputs(model_pattern)
        
        if not model_dirs:
            print("No model outputs found with predictions")
            return
            
        print(f"Found {len(model_dirs)} models to analyze")
        
        # Load analysis script if provided
        if analysis_script:
            analysis_path = Path(analysis_script)
            if not analysis_path.exists():
                analysis_path = self.project_dir / "templates" / analysis_script
        else:
            analysis_path = self.project_dir / "templates" / "fruitfly_alzheimers_analysis.py"
            
        print(f"Using analysis script: {analysis_path}")
            
        # Run analysis for each model
        for i, model_dir in enumerate(model_dirs, 1):
            print(f"\n[{i}/{len(model_dirs)}] Processing {model_dir.name}")
            result = self.run_analysis_for_model(model_dir, analysis_path)
            self.results.append(result)
            
        # Generate summary report
        self.generate_summary_report()
        
    def generate_summary_report(self):
        """Generate comprehensive summary of all analyses."""
        print("\n" + "="*60)
        print("Generating Analysis Summary Report")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create summary paths
        report_path = self.summary_dir / f"analysis_queue_{timestamp}_summary.md"
        csv_path = self.summary_dir / f"analysis_queue_{timestamp}_metrics.csv"
        # Skip JSON file - CSV and summary are more useful
        
        # Prepare data for CSV
        rows = []
        for result in self.results:
            if result["status"] == "completed":
                # Extract metrics from analysis output if available
                row = {
                    "model_name": result["model_name"],
                    "full_experiment_name": result.get("full_experiment_name", result["model_name"]),
                    "model_architecture": result.get("model_architecture", "Unknown"),
                    "target_variable": result.get("target_variable", "Unknown"),
                    "tissue_type": result.get("tissue_type", "Unknown"),
                    "cell_type": result.get("cell_type", "Unknown"),
                    "sex": result.get("sex", "Unknown"),
                    "comparison": result.get("comparison", "Unknown"),
                    "batch_corrected": result.get("batch_corrected", "Unknown"),
                    "sample_size": result.get("sample_size", "Unknown"),
                    "n_features": result.get("n_features", "Unknown"),
                    "epochs_trained": result.get("epochs_trained", "Unknown"),
                    "best_val_loss": result.get("best_val_loss", "Unknown"),
                    "n_predictions": result.get("n_predictions", 0),
                    "status": result["status"],
                    # Add model quality metrics
                    "model_quality": result.get("model_quality", "Unknown"),
                    "prediction_confidence": result.get("prediction_confidence", "Unknown"),
                    "quality_metric": result.get("quality_metric", "N/A"),
                    "control_auc": result.get("control_auc", 0),
                    "disease_auc": result.get("disease_auc", 0)
                }
                
                # Add performance metrics if available
                if "performance_metrics" in result:
                    perf = result["performance_metrics"]
                    row.update({
                        "accuracy": perf.get("accuracy"),
                        "f1_score": perf.get("f1_score"),
                        "auc": perf.get("auc"),
                        "precision": perf.get("precision"),
                        "recall": perf.get("recall"),
                        "mae": perf.get("mae"),
                        "rmse": perf.get("rmse"),
                        "r2_score": perf.get("r2_score")
                    })
                
                # Parse analysis result from output if available
                if "analysis_output" in result:
                    try:
                        # Look for ANALYSIS_RESULT: {...} in the output
                        output = result["analysis_output"]
                        if "ANALYSIS_RESULT:" in output:
                            json_start = output.find("ANALYSIS_RESULT:") + len("ANALYSIS_RESULT:")
                            json_str = output[json_start:].strip()
                            # Find the JSON object (may span multiple lines)
                            if json_str.startswith("{"):
                                brace_count = 0
                                end_pos = 0
                                for i, char in enumerate(json_str):
                                    if char == "{":
                                        brace_count += 1
                                    elif char == "}":
                                        brace_count -= 1
                                        if brace_count == 0:
                                            end_pos = i + 1
                                            break
                                
                                if end_pos > 0:
                                    analysis_result = json.loads(json_str[:end_pos])
                                    
                                    # Store n_predictions back to main result for detailed section
                                    if "n_predictions" in analysis_result:
                                        result["n_predictions"] = analysis_result["n_predictions"]
                                        row["n_predictions"] = analysis_result["n_predictions"]
                                    
                                    # Extract metrics if available
                                    if "metrics" in analysis_result:
                                        metrics = analysis_result["metrics"]
                                        row.update({
                                            "mean_error": metrics.get("mean_error"),
                                            "std_error": metrics.get("std_error"),
                                            "older_percentage": metrics.get("older_percentage"),
                                            "accelerated_aging": metrics.get("accelerated_aging"),
                                            "control_corrected": metrics.get("control_corrected")
                                        })
                                        
                                        # Add control baseline metrics if available
                                        if "true_acceleration" in metrics:
                                            row["true_acceleration"] = metrics["true_acceleration"]
                                    
                                    # Add control baseline info if available
                                    if "control_baseline" in analysis_result:
                                        control = analysis_result["control_baseline"]
                                        row.update({
                                            "control_experiment": control.get("control_experiment"),
                                            "control_mean_error": control.get("control_mean_error"),
                                            "control_std_error": control.get("control_std_error"),
                                            "control_n_samples": control.get("control_n_samples"),
                                            "control_interpretation": control.get("interpretation"),
                                            "statistical_significance": control.get("statistical_test", {}).get("significance"),
                                            "p_value": control.get("statistical_test", {}).get("p_value")
                                        })
                                    
                                    # Add genotype analysis if available
                                    if "genotype_analysis" in analysis_result:
                                        for genotype, data in analysis_result["genotype_analysis"].items():
                                            row[f"{genotype}_mean_error"] = data.get("mean_error")
                                            row[f"{genotype}_older_pct"] = data.get("older_percentage")
                    except Exception as e:
                        logger.warning(f"Could not parse analysis result for {result['model_name']}: {e}")
                
                rows.append(row)
        
        # Save CSV
        if rows:
            df = pd.DataFrame(rows)
            
            # Reorder columns for better readability
            column_order = [
                # Basic identification
                "model_name", "full_experiment_name", "status",
                # Model architecture and setup
                "model_architecture", "target_variable", "tissue_type", "cell_type", "sex",
                "comparison", "batch_corrected", 
                # Data characteristics
                "sample_size", "n_features", "n_predictions",
                # Training metrics
                "epochs_trained", "best_val_loss",
                # Performance metrics
                "accuracy", "f1_score", "auc", "precision", "recall",
                # Control baseline results
                "control_experiment", "control_mean_error", "control_std_error", "control_n_samples",
                # Disease results
                "mean_error", "std_error", "older_percentage", "accelerated_aging",
                # Comparison
                "true_acceleration", "control_interpretation", "statistical_significance", "p_value",
                "control_corrected"
            ]
            
            # Only include columns that exist in the dataframe
            existing_columns = [col for col in column_order if col in df.columns]
            # Add any remaining columns not in our order
            remaining_columns = [col for col in df.columns if col not in existing_columns]
            final_columns = existing_columns + remaining_columns
            
            df = df[final_columns]
            df.to_csv(csv_path, index=False)
            print(f"Metrics CSV saved: {csv_path}")
        
        # Skip JSON file - CSV and summary provide all needed information
        
        # Generate Markdown report
        with open(report_path, "w") as f:
            f.write("# Analysis Queue Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Models Analyzed:** {len(self.results)}\n\n")
            
            # Add interpretation guide
            f.write("## How to Interpret Results\n\n")
            f.write("### Key Metrics:\n")
            f.write("- **Control AUC**: How well the model predicts age on control data (0.75+ = reliable model)\n")
            f.write("- **Disease AUC**: Classification performance on disease data (may be low due to aging effects)\n") 
            f.write("- **True Acceleration**: Days of aging acceleration after removing model bias\n\n")
            
            f.write("### Evidence Levels (Based on True Acceleration):\n")
            f.write("- **Strong** (>2.0 days): Clear accelerated aging - disease flies appear >2 days older than controls\n")
            f.write("- **Moderate** (1.0-2.0 days): Moderate aging acceleration - disease flies appear 1-2 days older\n")
            f.write("- **Mild** (0.5-1.0 days): Slight aging acceleration - disease flies appear 0.5-1 day older\n") 
            f.write("- **None** (-0.5 to +0.5 days): No clear aging effect detected\n")
            f.write("- **Protective** (<-0.5 days): Disease condition appears protective - flies look younger than controls\n\n")
            
            f.write("### Understanding AUC Values:\n")
            f.write("- **High Control AUC + Low Disease AUC**: Model is good AND detects aging (disease flies look older)\n")
            f.write("- **High Control AUC + High Disease AUC**: Model is good but minimal aging detected\n")
            f.write("- **Low Control AUC**: Don't trust results - model can't predict age reliably\n\n")
            
            f.write("### Focus On:\n")
            f.write("- Models with **Control AUC ≥ 0.75** (reliable age prediction)\n")
            f.write("- **True Acceleration** values (control-corrected aging effect)\n")
            f.write("- **Strong & Moderate** results section for significant effects\n\n")
            
            # Summary statistics
            completed = [r for r in self.results if r["status"] == "completed"]
            failed = [r for r in self.results if r["status"] == "failed"]
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Completed:** {len(completed)}\n")
            f.write(f"- **Failed:** {len(failed)}\n\n")
            
            # Analysis results if available
            if rows:
                # Check if we have metrics data
                has_metrics_data = any("mean_error" in row and row["mean_error"] is not None for row in rows)
                
                if has_metrics_data:
                    f.write("## Aging Acceleration Results\n\n")
                    
                    # Sort by effect size if available
                    sorted_rows = sorted(rows, key=lambda x: x.get("mean_error", 0), reverse=True)
                    
                    f.write("| Full Experiment Name | Cell Type | Sex | Control AUC | Disease AUC | Model Quality | Mean Error | True Acceleration | Older % | Raw Evidence | Control-Corrected Evidence |\n")
                    f.write("|---------------------|-----------|-----|-------------|-------------|---------------|------------|------------------|---------|--------------|---------------------------|\n")
                    
                    for row in sorted_rows:
                        if row.get("mean_error") is not None:
                            mean_error = row["mean_error"]
                            older_pct = row.get("older_percentage", 0)
                            full_name = row.get("full_experiment_name", row["model_name"])
                            architecture = row.get("model_architecture", "Unknown")
                            cell_type = row.get("cell_type", "Unknown")
                            sex = row.get("sex", "Unknown")
                            
                            # Raw evidence (based on mean error)
                            raw_evidence = "Strong" if mean_error > 2.0 else \
                                          "Moderate" if mean_error > 1.0 else \
                                          "Weak" if mean_error > 0 else "None"
                            
                            # Control-corrected evidence (based on true acceleration if available)
                            true_acceleration = row.get("true_acceleration", mean_error)
                            corrected_evidence = "Strong" if true_acceleration > 2.0 else \
                                               "Moderate" if true_acceleration > 1.0 else \
                                               "Mild" if true_acceleration > 0.5 else \
                                               "None" if true_acceleration > -0.5 else "Protective"
                            
                            # Get model quality info
                            control_auc = row.get("control_auc", 0)
                            disease_auc = row.get("disease_auc", 0)
                            model_quality = row.get("model_quality", "Unknown")
                            
                            # Format AUC values
                            control_auc_str = f"{control_auc:.3f}" if control_auc > 0 else "N/A"
                            disease_auc_str = f"{disease_auc:.3f}" if disease_auc > 0 else "N/A"
                            
                            f.write(
                                f"| {full_name} | "
                                f"{cell_type} | "
                                f"{sex} | "
                                f"{control_auc_str} | "
                                f"{disease_auc_str} | "
                                f"{model_quality} | "
                                f"{mean_error:+.2f} | "
                                f"{true_acceleration:+.2f} | "
                                f"{older_pct:.1f}% | "
                                f"{raw_evidence} | "
                                f"{corrected_evidence} |\n"
                            )
                    
                    f.write("\n")
                    
                    # Add strong/moderate results section
                    strong_moderate_rows = []
                    for r in sorted_rows:
                        if r.get("mean_error") is not None:
                            true_accel = r.get("true_acceleration", r["mean_error"])
                            if true_accel > 1.0:  # Moderate or stronger acceleration
                                strong_moderate_rows.append(r)
                    
                    if strong_moderate_rows:
                        f.write("### Strong & Moderate Aging Acceleration Results\n\n")
                        f.write("| Cell Type | Sex | Control AUC | Disease AUC | Mean Error | True Acceleration | Evidence | Interpretation |\n")
                        f.write("|-----------|-----|-------------|-------------|------------|------------------|----------|----------------|\n")
                        
                        for row in strong_moderate_rows:
                            cell_type = row.get("cell_type", "Unknown")
                            sex = row.get("sex", "Unknown")
                            control_auc = row.get("control_auc", 0)
                            disease_auc = row.get("disease_auc", 0)
                            mean_error = row["mean_error"]
                            true_acceleration = row.get("true_acceleration", mean_error)
                            
                            # Determine interpretation
                            if true_acceleration > 2.0:
                                interpretation = "Strong aging acceleration"
                            elif true_acceleration > 1.0:
                                interpretation = "Moderate aging acceleration"
                            elif true_acceleration > 0.5:
                                interpretation = "Mild aging acceleration"
                            elif true_acceleration > -0.5:
                                interpretation = "No clear effect"
                            else:
                                interpretation = "Protective effect"
                            
                            corrected_evidence = "Strong" if true_acceleration > 2.0 else \
                                               "Moderate" if true_acceleration > 1.0 else \
                                               "Mild" if true_acceleration > 0.5 else \
                                               "None" if true_acceleration > -0.5 else "Protective"
                            
                            # Format AUC values
                            control_auc_str = f"{control_auc:.3f}" if control_auc > 0 else "N/A"
                            disease_auc_str = f"{disease_auc:.3f}" if disease_auc > 0 else "N/A"
                            
                            f.write(
                                f"| {cell_type} | {sex} | {control_auc_str} | {disease_auc_str} | "
                                f"{mean_error:+.2f} | {true_acceleration:+.2f} | "
                                f"{corrected_evidence} | {interpretation} |\n"
                            )
                        f.write("\n")
                    
                    # Overall statistics
                    all_errors = [row["mean_error"] for row in rows if row.get("mean_error") is not None]
                    if all_errors:
                        overall_mean = sum(all_errors) / len(all_errors)
                        f.write("## Overall Findings\n\n")
                        f.write(f"- **Average effect across all models:** {overall_mean:+.2f}\n")
                        f.write(f"- **Models showing positive effect:** {sum(1 for e in all_errors if e > 0)}/{len(all_errors)}\n")
                        f.write(f"- **Models with strong evidence:** {sum(1 for e in all_errors if e > 2.0)}/{len(all_errors)}\n\n")
            
            # Detailed results
            f.write("## Detailed Model Information\n\n")
            if rows:
                # Create a comprehensive table for completed models
                completed_models = [r for r in rows if r.get("mean_error") is not None]
                if completed_models:
                    f.write("### Model Architecture and Performance Summary\n\n")
                    f.write("| Experiment | Architecture | Tissue | Cell Type | Sex | Samples | Features | Epochs | Val Loss | Accuracy | AUC | Mean Error | True Acceleration | Raw Evidence | Control-Corrected Evidence |\n")
                    f.write("|------------|-------------|--------|-----------|-----|---------|----------|--------|----------|----------|-----|------------|------------------|--------------|---------------------------|\n")
                    
                    for row in sorted(completed_models, key=lambda x: x.get("mean_error", 0), reverse=True):
                        mean_error = row["mean_error"]
                        full_name = row.get("full_experiment_name", row["model_name"])
                        architecture = row.get("model_architecture", "N/A")
                        tissue = row.get("tissue_type", "N/A")
                        cell_type = row.get("cell_type", "N/A")
                        sex = row.get("sex", "N/A")
                        samples = row.get("sample_size", "N/A")
                        features = row.get("n_features", "N/A")
                        epochs = row.get("epochs_trained", "N/A")
                        val_loss = row.get("best_val_loss", "N/A")
                        accuracy = row.get("accuracy", "N/A")
                        auc = row.get("auc", "N/A")
                        
                        # Calculate both evidence types
                        raw_evidence = "Strong" if mean_error > 2.0 else \
                                      "Moderate" if mean_error > 1.0 else \
                                      "Weak" if mean_error > 0 else "None"
                        
                        true_acceleration_val = row.get("true_acceleration", mean_error)
                        corrected_evidence = "Strong" if true_acceleration_val > 2.0 else \
                                           "Moderate" if true_acceleration_val > 1.0 else \
                                           "Mild" if true_acceleration_val > 0.5 else \
                                           "None" if true_acceleration_val > -0.5 else "Protective"
                        
                        # Format numerical values
                        if isinstance(val_loss, float):
                            val_loss = f"{val_loss:.3f}"
                        if isinstance(accuracy, float):
                            accuracy = f"{accuracy:.3f}"
                        if isinstance(auc, float):
                            auc = f"{auc:.3f}"
                        
                        f.write(
                            f"| {full_name} | {architecture} | {tissue} | {cell_type} | {sex} | "
                            f"{samples} | {features} | {epochs} | {val_loss} | {accuracy} | {auc} | "
                            f"{mean_error:+.2f} | {true_acceleration_val:+.2f} | {raw_evidence} | {corrected_evidence} |\n"
                        )
                    f.write("\n")
                    
                    # Add control baseline comparison section
                    f.write("### Control Baseline Comparisons\n\n")
                    control_rows = [r for r in completed_models if r.get("control_experiment")]
                    if control_rows:
                        f.write("| Experiment | Control Baseline | Control Error | Disease Error | Net Effect | P-value | Significance |\n")
                        f.write("|------------|------------------|---------------|---------------|------------|---------|-------------|\n")
                        
                        for row in control_rows:
                            full_name = row.get("full_experiment_name", row["model_name"])
                            control_exp = row.get("control_experiment", "N/A")
                            control_error = row.get("control_mean_error", "N/A")
                            disease_error = row.get("mean_error", "N/A")
                            net_effect = row.get("true_acceleration", "N/A")
                            p_value = row.get("p_value", "N/A")
                            significance = row.get("statistical_significance", "N/A")
                            
                            # Format numerical values
                            if isinstance(control_error, float):
                                control_error = f"{control_error:+.2f}"
                            if isinstance(disease_error, float):
                                disease_error = f"{disease_error:+.2f}"
                            if isinstance(net_effect, float):
                                net_effect = f"{net_effect:+.2f}"
                            if isinstance(p_value, float):
                                p_value = f"{p_value:.3f}"
                            
                            f.write(
                                f"| {full_name} | {control_exp} | {control_error} | {disease_error} | "
                                f"{net_effect} | {p_value} | {significance} |\n"
                            )
                        f.write("\n")
            
            # Show failed/problematic models in separate section
            failed_models = [r for r in self.results if r["status"] != "completed"]
            if failed_models:
                f.write("### Failed/Incomplete Models\n\n")
                f.write("| Model | Status | Issue |\n")
                f.write("|-------|--------|-------|\n")
                
                for result in failed_models:
                    status_icon = "❌" if result["status"] == "failed" else "⚠️"
                    issue = result.get("error", "No predictions found" if result["status"] == "no_predictions" else "No analysis script")
                    f.write(f"| {result['model_name']} | {status_icon} {result['status']} | {issue} |\n")
                f.write("\n")
        
        print(f"Summary report saved: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        
        if rows:
            models_with_metrics = [r for r in rows if r.get("mean_error") is not None]
            if models_with_metrics:
                all_errors = [r["mean_error"] for r in models_with_metrics]
                overall_mean = sum(all_errors) / len(all_errors)
                
                print(f"Average effect: {overall_mean:+.2f}")
                print(f"Models showing positive effect: {sum(1 for e in all_errors if e > 0)}/{len(all_errors)}")
                
                # Show top 3
                sorted_models = sorted(models_with_metrics, key=lambda x: x["mean_error"], reverse=True)[:3]
                print("\nTop 3 Models (Strongest Effect):")
                for i, model in enumerate(sorted_models, 1):
                    print(f"  {i}. {model['model_name']}: {model['mean_error']:+.2f}")
        
        print(f"\nAll results saved in: {self.summary_dir}")


def main():
    """Main entry point for the analysis queue runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis queue on multiple models")
    parser.add_argument(
        "--pattern",
        default="*",
        help="Pattern to match model directories (default: *)"
    )
    parser.add_argument(
        "--analysis-script",
        help="Path to analysis script (default: templates/fruitfly_alzheimers_analysis.py)"
    )
    
    args = parser.parse_args()
    
    runner = AnalysisQueueRunner()
    runner.run_queue(
        model_pattern=args.pattern,
        analysis_script=args.analysis_script
    )


if __name__ == "__main__":
    main()