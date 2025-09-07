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
        self.summary_dir = self.project_dir / "outputs" / "analysis_summaries"
        self.summary_dir.mkdir(parents=True, exist_ok=True)
        
    def run_analysis_for_model(self, model_dir: Path, analysis_script: Path) -> Dict:
        """Run analysis script for a single model."""
        result = {
            "model_name": model_dir.name,
            "model_path": str(model_dir),
            "status": "pending",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            print(f"\n{'='*60}")
            print(f"Running analysis for: {model_dir.name}")
            print(f"{'='*60}")
            
            # Check if predictions exist
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
            if predictions_file.parent.name == "evaluation":
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
        json_path = self.summary_dir / f"analysis_queue_{timestamp}_full.json"
        
        # Prepare data for CSV
        rows = []
        for result in self.results:
            if result["status"] == "completed":
                # Extract metrics from analysis output if available
                row = {
                    "model_name": result["model_name"],
                    "n_predictions": result.get("n_predictions", 0),
                    "status": result["status"]
                }
                
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
                                            "control_interpretation": control.get("interpretation"),
                                            "statistical_significance": control.get("statistical_test", {}).get("significance")
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
            df.to_csv(csv_path, index=False)
            print(f"Metrics CSV saved: {csv_path}")
        
        # Save full JSON results
        with open(json_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"Full results saved: {json_path}")
        
        # Generate Markdown report
        with open(report_path, "w") as f:
            f.write("# Analysis Queue Summary\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Models Analyzed:** {len(self.results)}\n\n")
            
            # Summary statistics
            completed = [r for r in self.results if r["status"] == "completed"]
            failed = [r for r in self.results if r["status"] == "failed"]
            no_predictions = [r for r in self.results if r["status"] == "no_predictions"]
            no_script = [r for r in self.results if r["status"] == "no_script"]
            
            f.write("## Summary Statistics\n\n")
            f.write(f"- **Completed:** {len(completed)}\n")
            f.write(f"- **Failed:** {len(failed)}\n")
            f.write(f"- **No Predictions:** {len(no_predictions)}\n")
            f.write(f"- **No Script:** {len(no_script)}\n\n")
            
            # Analysis results if available
            if rows:
                # Check if we have metrics data
                has_metrics_data = any("mean_error" in row and row["mean_error"] is not None for row in rows)
                
                if has_metrics_data:
                    f.write("## Analysis Results\n\n")
                    
                    # Sort by effect size if available
                    sorted_rows = sorted(rows, key=lambda x: x.get("mean_error", 0), reverse=True)
                    
                    f.write("| Model | Mean Error | Older % | Evidence Level |\n")
                    f.write("|-------|------------|---------|----------------|\n")
                    
                    for row in sorted_rows:
                        if row.get("mean_error") is not None:
                            mean_error = row["mean_error"]
                            older_pct = row.get("older_percentage", 0)
                            
                            evidence = "Strong" if mean_error > 2.0 else \
                                      "Moderate" if mean_error > 1.0 else \
                                      "Weak" if mean_error > 0 else "None"
                            
                            f.write(
                                f"| {row['model_name'][:30]} | "
                                f"{mean_error:+.2f} | "
                                f"{older_pct:.1f}% | "
                                f"{evidence} |\n"
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
            f.write("## Detailed Results\n\n")
            for result in self.results:
                f.write(f"### {result['model_name']}\n\n")
                f.write(f"- **Status:** {result['status']}\n")
                f.write(f"- **Path:** {result['model_path']}\n")
                
                if result["status"] == "completed":
                    f.write(f"- **Predictions analyzed:** {result.get('n_predictions', 'N/A')}\n")
                elif result["status"] == "failed":
                    f.write(f"- **Error:** {result.get('error', 'Unknown error')}\n")
                    
                f.write("\n")
        
        print(f"Summary report saved: {report_path}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"Completed: {len(completed)}")
        print(f"Failed: {len(failed)}")
        print(f"No Predictions: {len(no_predictions)}")
        print(f"No Script: {len(no_script)}")
        
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