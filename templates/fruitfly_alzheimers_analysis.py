"""
TimeFlies Alzheimer's Aging Acceleration Analysis

Project-specific analysis for fruitfly_alzheimers project.
Analyzes prediction results to determine if Alzheimer's flies are predicted
as older than their chronological age, indicating accelerated aging.

Usage:
- Automatically used when project = "fruitfly_alzheimers"
- Or explicitly: timeflies analyze --analysis-script templates/fruitfly_alzheimers_analysis.py
"""

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def run_analysis(model, config, path_manager, pipeline):
    """
    Main analysis function for Alzheimer's aging acceleration.

    This function will be called by TimeFlies with the pipeline context.
    """
    logger.debug("Starting Alzheimer's Aging Acceleration Analysis...")

    try:
        print("\n🧬 Alzheimer's Aging Acceleration Analysis")
        print("=" * 60)

        # Try to find the predictions file
        predictions_file = None
        experiment_dir = None

        # First check if the pipeline has a specific predictions path (from CLI)
        if hasattr(pipeline, "_analysis_predictions_path"):
            predictions_file = Path(pipeline._analysis_predictions_path)
            # For best model paths, the experiment dir is parent.parent
            # This handles both regular and best model paths correctly
            experiment_dir = predictions_file.parent.parent
            print("🎯 Using CLI-specified predictions from best model")
            logger.debug(f"Using predictions from: {predictions_file}")
            logger.debug(f"Using experiment dir: {experiment_dir}")
        else:
            # Try to get experiment directory from pipeline
            try:
                experiment_dir = path_manager.get_experiment_dir(
                    getattr(pipeline, "experiment_name", None)
                )
                predictions_file = (
                    Path(experiment_dir) / "evaluation" / "predictions.csv"
                )
            except Exception:
                experiment_dir = None

        # If not found, search for predictions from the best model
        if not predictions_file or not predictions_file.exists():
            # Look for predictions in the best model directory first
            base_path = Path(path_manager.base_path) / "experiments"

            # Try to find best model predictions first
            best_pattern = "**/best/**/evaluation/predictions.csv"
            best_prediction_files = list(base_path.glob(best_pattern))

            if best_prediction_files:
                # Use the most recent best model predictions for this config
                predictions_file = max(
                    best_prediction_files, key=lambda x: x.stat().st_mtime
                )
                experiment_dir = predictions_file.parent.parent
                print("🏆 Using predictions from best model for this configuration")
            else:
                # Fallback to any predictions file
                search_pattern = "**/evaluation/predictions.csv"
                prediction_files = list(base_path.glob(search_pattern))
                if prediction_files:
                    predictions_file = max(
                        prediction_files, key=lambda x: x.stat().st_mtime
                    )
                    experiment_dir = predictions_file.parent.parent
                    print(f"📍 Using most recent predictions: {predictions_file}")
                else:
                    print("❌ No predictions found. Run evaluation first.")
                    return

        # Create analysis directory in the same experiment directory
        analysis_dir = Path(experiment_dir) / "alzheimers_analysis"

        # Ensure the parent directory exists (important for best model paths)
        analysis_dir.parent.mkdir(parents=True, exist_ok=True)
        analysis_dir.mkdir(exist_ok=True)

        logger.debug(f"Created analysis directory at: {analysis_dir}")

        df = pd.read_csv(predictions_file)
        print(f"📊 Loaded {len(df)} predictions")

        # Check for required columns
        required_cols = ["actual_age", "predicted_age"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Missing required columns: {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
            return

        # Add prediction error column
        df["prediction_error"] = pd.to_numeric(df["predicted_age"]) - pd.to_numeric(
            df["actual_age"]
        )

        # Run analyses
        aging_acceleration_analysis(df, analysis_dir)
        if "genotype" in df.columns:
            genotype_comparison_analysis(df, analysis_dir)
        age_specific_analysis(df, analysis_dir)
        create_comprehensive_visualizations(df, analysis_dir)
        generate_summary_report(df, analysis_dir)

        print("✅ Alzheimer's analysis completed!")
        print(f"📁 Results saved in: {analysis_dir}")

    except Exception as e:
        logger.error(f"Alzheimer's analysis failed: {e}")
        raise


def aging_acceleration_analysis(df, output_dir):
    """Analyze overall aging acceleration patterns."""
    print("\n1️⃣ Aging Acceleration Analysis")

    # Basic statistics
    mean_error = df["prediction_error"].mean()
    std_error = df["prediction_error"].std()

    print(f"   Mean prediction error: {mean_error:.2f} days")
    print(f"   Std prediction error: {std_error:.2f} days")

    if mean_error > 0:
        print(f"   ✅ Flies predicted {mean_error:.2f} days OLDER (accelerated aging)")
    elif mean_error < 0:
        print(f"   ⚠️ Flies predicted {abs(mean_error):.2f} days YOUNGER")
    else:
        print("   ➖ No aging bias detected")

    # Age distribution analysis
    if "actual_age" in df.columns:
        age_counts = df["actual_age"].value_counts().sort_index()
        print(f"   Age distribution: {age_counts.to_dict()}")

    # Save basic results
    basic_stats = {
        "mean_prediction_error": float(mean_error),
        "std_prediction_error": float(std_error),
        "n_samples": len(df),
        "accelerated_aging": mean_error > 0,
    }

    import json

    with open(output_dir / "basic_aging_stats.json", "w") as f:
        json.dump(basic_stats, f, indent=2)


def genotype_comparison_analysis(df, output_dir):
    """Analyze aging acceleration by genotype (if genotype data available)."""
    print("\n2️⃣ Genotype Comparison Analysis")

    try:
        # Separate by genotype
        genotype_stats = []

        for genotype in df["genotype"].unique():
            genotype_data = df[df["genotype"] == genotype]
            mean_error = genotype_data["prediction_error"].mean()
            std_error = genotype_data["prediction_error"].std()
            n_samples = len(genotype_data)
            older_pct = (genotype_data["prediction_error"] > 0).mean() * 100

            genotype_stats.append(
                {
                    "genotype": genotype,
                    "mean_error": float(mean_error),
                    "std_error": float(std_error),
                    "n_samples": int(n_samples),
                    "older_percentage": float(older_pct),
                }
            )

            print(
                f"   {genotype}: {mean_error:+.2f} ± {std_error:.2f} days (n={n_samples}, {older_pct:.1f}% older)"
            )

        # Statistical comparisons between genotypes
        genotypes = list(df["genotype"].unique())
        if len(genotypes) >= 2:
            print("\n   Statistical comparisons:")
            for i, gen1 in enumerate(genotypes[:-1]):
                for gen2 in genotypes[i + 1 :]:
                    data1 = df[df["genotype"] == gen1]["prediction_error"]
                    data2 = df[df["genotype"] == gen2]["prediction_error"]
                    if len(data1) > 0 and len(data2) > 0:
                        statistic, p_value = stats.ttest_ind(data1, data2)
                        significance = (
                            "***"
                            if p_value < 0.001
                            else "**"
                            if p_value < 0.01
                            else "*"
                            if p_value < 0.05
                            else "ns"
                        )
                        print(f"   {gen1} vs {gen2}: p={p_value:.2e} {significance}")

        # Save genotype results
        genotype_df = pd.DataFrame(genotype_stats)
        genotype_df.to_csv(output_dir / "genotype_comparison.csv", index=False)

    except Exception as e:
        logger.error(f"Genotype comparison failed: {e}")


def age_specific_analysis(df, output_dir):
    """Analyze prediction errors by age group."""
    print("\n3️⃣ Age-Specific Analysis")

    try:
        age_stats = []

        for age in sorted(df["actual_age"].unique()):
            age_data = df[df["actual_age"] == age]
            mean_error = age_data["prediction_error"].mean()
            std_error = age_data["prediction_error"].std()
            n_samples = len(age_data)

            age_stats.append(
                {
                    "age": int(age),
                    "mean_error": float(mean_error),
                    "std_error": float(std_error),
                    "n_samples": int(n_samples),
                }
            )

            print(
                f"   Age {age}: {mean_error:+.2f} ± {std_error:.2f} days (n={n_samples})"
            )

        # Save age-specific results
        age_df = pd.DataFrame(age_stats)
        age_df.to_csv(output_dir / "age_specific_errors.csv", index=False)

    except Exception as e:
        logger.error(f"Age-specific analysis failed: {e}")


def create_comprehensive_visualizations(df, output_dir):
    """Create comprehensive visualizations."""
    print("\n4️⃣ Creating Visualizations")

    try:
        # Determine figure layout based on available data
        has_genotype = "genotype" in df.columns and len(df["genotype"].unique()) > 1

        fig, axes = plt.subplots(
            2, 2 if has_genotype else 2, figsize=(15, 12) if has_genotype else (12, 12)
        )
        if not has_genotype:
            axes = np.append(axes, [None])  # Pad for consistent indexing

        # 1. Prediction error distribution
        axes[0, 0].hist(
            df["prediction_error"],
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        axes[0, 0].axvline(0, color="red", linestyle="--", label="Perfect Prediction")
        axes[0, 0].axvline(
            df["prediction_error"].mean(),
            color="orange",
            linestyle="-",
            label=f"Mean = {df['prediction_error'].mean():.2f}",
        )
        axes[0, 0].set_xlabel("Prediction Error (Predicted - Actual Age)")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].set_title("Aging Acceleration Distribution")
        axes[0, 0].legend()

        # 2. Actual vs Predicted scatter
        axes[0, 1].scatter(
            df["actual_age"], df["predicted_age"], alpha=0.6, color="blue"
        )
        min_age, max_age = df["actual_age"].min(), df["actual_age"].max()
        axes[0, 1].plot(
            [min_age, max_age],
            [min_age, max_age],
            "r--",
            alpha=0.5,
            label="Perfect Prediction",
        )
        axes[0, 1].set_xlabel("Actual Age (days)")
        axes[0, 1].set_ylabel("Predicted Age (days)")
        axes[0, 1].set_title("Actual vs Predicted Age")
        axes[0, 1].legend()

        # 3. Error by age
        age_means = df.groupby("actual_age")["prediction_error"].mean()
        age_stds = df.groupby("actual_age")["prediction_error"].std()
        axes[1, 0].errorbar(
            age_means.index,
            age_means.values,
            yerr=age_stds.values,
            fmt="o-",
            capsize=5,
            capthick=2,
            color="green",
        )
        axes[1, 0].axhline(0, color="red", linestyle="--", alpha=0.5)
        axes[1, 0].set_xlabel("Actual Age (days)")
        axes[1, 0].set_ylabel("Mean Prediction Error (days)")
        axes[1, 0].set_title("Prediction Error by Age")

        # 4. Genotype comparison (if available)
        if has_genotype:
            genotype_data = []
            genotype_labels = []

            for genotype in df["genotype"].unique():
                genotype_errors = df[df["genotype"] == genotype]["prediction_error"]
                genotype_data.append(genotype_errors)
                genotype_labels.append(f"{genotype}\n(n={len(genotype_errors)})")

            axes[1, 1].boxplot(genotype_data, labels=genotype_labels)
            axes[1, 1].axhline(0, color="red", linestyle="--", alpha=0.5)
            axes[1, 1].set_ylabel("Prediction Error (days)")
            axes[1, 1].set_title("Prediction Error by Genotype")
            axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()

        # Save plot
        output_path = output_dir / "alzheimers_comprehensive_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"   Visualization saved: {output_path}")

    except Exception as e:
        logger.error(f"Visualization creation failed: {e}")


def generate_summary_report(df, output_dir):
    """Generate comprehensive summary report."""
    print("\n5️⃣ Generating Summary Report")

    try:
        # Calculate key metrics
        mean_error = df["prediction_error"].mean()
        std_error = df["prediction_error"].std()
        older_pct = (df["prediction_error"] > 0).mean() * 100

        # Create summary
        summary = {
            "analysis_type": "alzheimers_aging_acceleration",
            "timestamp": pd.Timestamp.now().isoformat(),
            "dataset_size": len(df),
            "mean_prediction_error_days": float(mean_error),
            "std_prediction_error_days": float(std_error),
            "percentage_predicted_older": float(older_pct),
            "accelerated_aging_evidence": mean_error
            > 1.0,  # More than 1 day older on average
            "interpretation": {
                "aging_acceleration": "Strong evidence"
                if mean_error > 2.0
                else "Moderate evidence"
                if mean_error > 1.0
                else "Weak evidence"
                if mean_error > 0
                else "No evidence",
                "biological_significance": "Model detects disrupted aging patterns in Alzheimer's flies"
                if mean_error > 1.0
                else "No clear aging acceleration detected",
            },
        }

        # Add genotype-specific results if available
        if "genotype" in df.columns:
            genotype_summary = {}
            for genotype in df["genotype"].unique():
                genotype_data = df[df["genotype"] == genotype]
                genotype_summary[genotype] = {
                    "n_samples": len(genotype_data),
                    "mean_error": float(genotype_data["prediction_error"].mean()),
                    "older_percentage": float(
                        (genotype_data["prediction_error"] > 0).mean() * 100
                    ),
                }
            summary["genotype_analysis"] = genotype_summary

        # Save summary
        import json

        with open(output_dir / "analysis_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Print key findings
        print("\n📋 KEY FINDINGS:")
        print(f"   Dataset: {len(df)} predictions")
        print(f"   Mean aging acceleration: {mean_error:+.2f} ± {std_error:.2f} days")
        print(f"   Flies predicted as older: {older_pct:.1f}%")
        print(f"   Evidence level: {summary['interpretation']['aging_acceleration']}")
        print(
            f"   Biological interpretation: {summary['interpretation']['biological_significance']}"
        )

        print(f"\n📄 Summary saved: {output_dir / 'analysis_summary.json'}")

    except Exception as e:
        logger.error(f"Summary report generation failed: {e}")


# For backward compatibility if called directly
class AlzheimersAgingAccelerationAnalyzer:
    """Legacy class wrapper for backward compatibility."""

    def __init__(self, model=None, config=None, path_manager=None):
        self.model = model
        self.config = config
        self.path_manager = path_manager

    def run_complete_analysis(self):
        """Run the analysis using the new interface."""

        # Create a mock pipeline object
        class MockPipeline:
            def __init__(self):
                self.experiment_name = None

        pipeline = MockPipeline()
        run_analysis(self.model, self.config, self.path_manager, pipeline)
