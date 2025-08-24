"""
TimeFlies Aging-Specific Analysis Template

Template for analyzing aging patterns in single-cell data.
Focus on age prediction, aging acceleration, and temporal patterns.

Usage:
timeflies analyze --analysis-script templates/aging_analysis_template.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)


def run_analysis(model, config, path_manager, pipeline):
    """
    Run aging-specific analysis.
    """
    logger.info("Starting aging-specific analysis...")

    try:
        # Setup
        project = getattr(config, 'project', 'unknown')
        tissue = config.data.tissue
        experiment_dir = path_manager.get_experiment_dir(getattr(pipeline, 'experiment_name', None))

        print(f"🔬 Aging Analysis for {project} - {tissue}")
        print("=" * 50)

        # Load predictions
        predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
        if not predictions_file.exists():
            print("❌ No predictions found. Run evaluation first.")
            return

        predictions_df = pd.read_csv(predictions_file)
        print(f"📊 Loaded {len(predictions_df)} predictions")

        # Create analysis directory
        analysis_dir = Path(experiment_dir) / "aging_analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Run aging-specific analyses
        aging_acceleration_analysis(predictions_df, analysis_dir)
        age_group_performance_analysis(predictions_df, analysis_dir)
        temporal_pattern_analysis(predictions_df, analysis_dir, pipeline)
        aging_biomarker_analysis(predictions_df, analysis_dir, pipeline)

        print("✅ Aging analysis completed!")
        print(f"📁 Results saved in: {analysis_dir}")

    except Exception as e:
        logger.error(f"Aging analysis failed: {e}")
        raise


def aging_acceleration_analysis(predictions_df, output_dir):
    """Analyze aging acceleration patterns."""
    print("\n1️⃣ Aging Acceleration Analysis")

    try:
        if 'true_age' not in predictions_df.columns or 'predicted_age' not in predictions_df.columns:
            print("   ⚠️ Age columns not found, skipping aging acceleration analysis")
            return

        # Calculate aging acceleration
        predictions_df['aging_acceleration'] = predictions_df['predicted_age'] - predictions_df['true_age']

        # Basic statistics
        acc_stats = {
            'mean_acceleration': predictions_df['aging_acceleration'].mean(),
            'std_acceleration': predictions_df['aging_acceleration'].std(),
            'accelerated_samples': (predictions_df['aging_acceleration'] > 0).sum(),
            'decelerated_samples': (predictions_df['aging_acceleration'] < 0).sum(),
        }

        print(f"   Mean aging acceleration: {acc_stats['mean_acceleration']:.2f}")
        print(f"   Std aging acceleration: {acc_stats['std_acceleration']:.2f}")
        print(f"   Accelerated samples: {acc_stats['accelerated_samples']} ({acc_stats['accelerated_samples']/len(predictions_df)*100:.1f}%)")
        print(f"   Decelerated samples: {acc_stats['decelerated_samples']} ({acc_stats['decelerated_samples']/len(predictions_df)*100:.1f}%)")

        # Plot aging acceleration distribution
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(predictions_df['aging_acceleration'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(0, color='red', linestyle='--', label='Perfect Prediction')
        plt.xlabel('Aging Acceleration (Predicted - True Age)')
        plt.ylabel('Count')
        plt.title('Aging Acceleration Distribution')
        plt.legend()

        plt.subplot(1, 3, 2)
        plt.scatter(predictions_df['true_age'], predictions_df['predicted_age'], alpha=0.6, color='blue')
        plt.plot([predictions_df['true_age'].min(), predictions_df['true_age'].max()],
                [predictions_df['true_age'].min(), predictions_df['true_age'].max()],
                'r--', label='Perfect Prediction')
        plt.xlabel('True Age')
        plt.ylabel('Predicted Age')
        plt.title('Age Prediction Accuracy')
        plt.legend()

        plt.subplot(1, 3, 3)
        plt.scatter(predictions_df['true_age'], predictions_df['aging_acceleration'], alpha=0.6, color='green')
        plt.axhline(0, color='red', linestyle='--', label='No Acceleration')
        plt.xlabel('True Age')
        plt.ylabel('Aging Acceleration')
        plt.title('Acceleration by Age')
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_dir / 'aging_acceleration.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Save results
        import json
        with open(output_dir / 'aging_acceleration_stats.json', 'w') as f:
            json.dump(acc_stats, f, indent=2)

        predictions_df.to_csv(output_dir / 'predictions_with_acceleration.csv', index=False)

    except Exception as e:
        logger.error(f"Aging acceleration analysis failed: {e}")


def age_group_performance_analysis(predictions_df, output_dir):
    """Analyze model performance across different age groups."""
    print("\n2️⃣ Age Group Performance Analysis")

    try:
        if 'true_age' not in predictions_df.columns:
            print("   ⚠️ True age not found, skipping age group analysis")
            return

        # Create age groups
        age_bins = pd.qcut(predictions_df['true_age'], q=4, labels=['Young', 'Middle-Young', 'Middle-Old', 'Old'])
        predictions_df['age_group'] = age_bins

        # Calculate metrics per age group
        group_stats = []
        for group in age_bins.categories:
            group_data = predictions_df[predictions_df['age_group'] == group]
            if len(group_data) > 0 and 'predicted_age' in predictions_df.columns:
                mae = np.mean(np.abs(group_data['true_age'] - group_data['predicted_age']))
                rmse = np.sqrt(np.mean((group_data['true_age'] - group_data['predicted_age'])**2))
                r2 = stats.pearsonr(group_data['true_age'], group_data['predicted_age'])[0]**2

                group_stats.append({
                    'age_group': group,
                    'n_samples': len(group_data),
                    'age_range': f"{group_data['true_age'].min():.1f}-{group_data['true_age'].max():.1f}",
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2
                })

                print(f"   {group}: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.3f} (n={len(group_data)})")

        # Save group statistics
        group_stats_df = pd.DataFrame(group_stats)
        group_stats_df.to_csv(output_dir / 'age_group_performance.csv', index=False)

        # Plot performance by age group
        if group_stats:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))

            axes[0].bar(group_stats_df['age_group'], group_stats_df['mae'], color='lightcoral')
            axes[0].set_title('MAE by Age Group')
            axes[0].set_ylabel('Mean Absolute Error')
            axes[0].tick_params(axis='x', rotation=45)

            axes[1].bar(group_stats_df['age_group'], group_stats_df['rmse'], color='lightblue')
            axes[1].set_title('RMSE by Age Group')
            axes[1].set_ylabel('Root Mean Square Error')
            axes[1].tick_params(axis='x', rotation=45)

            axes[2].bar(group_stats_df['age_group'], group_stats_df['r2'], color='lightgreen')
            axes[2].set_title('R² by Age Group')
            axes[2].set_ylabel('R-squared')
            axes[2].tick_params(axis='x', rotation=45)

            plt.tight_layout()
            plt.savefig(output_dir / 'age_group_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

    except Exception as e:
        logger.error(f"Age group performance analysis failed: {e}")


def temporal_pattern_analysis(predictions_df, output_dir, pipeline):
    """Analyze temporal patterns in aging data."""
    print("\n3️⃣ Temporal Pattern Analysis")

    try:
        # This would analyze batch effects, time-series patterns, etc.
        # For now, a placeholder showing how to access the data
        if hasattr(pipeline, 'adata_eval') and pipeline.adata_eval is not None:
            adata = pipeline.adata_eval
            print(f"   Evaluation data: {adata.n_obs} cells, {adata.n_vars} genes")

            # Check for temporal metadata
            temporal_cols = [col for col in adata.obs.columns if 'time' in col.lower() or 'batch' in col.lower() or 'date' in col.lower()]
            if temporal_cols:
                print(f"   Found temporal columns: {temporal_cols}")
            else:
                print("   No obvious temporal columns found")

        print("   Temporal analysis placeholder - extend based on your data structure")

    except Exception as e:
        logger.error(f"Temporal pattern analysis failed: {e}")


def aging_biomarker_analysis(predictions_df, output_dir, pipeline):
    """Analyze potential aging biomarkers."""
    print("\n4️⃣ Aging Biomarker Analysis")

    try:
        # Placeholder for biomarker analysis
        # This would typically involve:
        # 1. Feature importance from the model
        # 2. Correlation with aging acceleration
        # 3. Gene set enrichment analysis
        # 4. Pathway analysis

        if hasattr(pipeline, 'adata_eval') and pipeline.adata_eval is not None:
            adata = pipeline.adata_eval
            n_genes = adata.n_vars
            print(f"   Available for biomarker analysis: {n_genes} genes")

            # Example: Top variable genes
            if 'highly_variable' in adata.var.columns:
                hvg_count = adata.var['highly_variable'].sum()
                print(f"   Highly variable genes: {hvg_count}")

        print("   Biomarker analysis placeholder - extend based on your analysis needs")

        # Save a placeholder results file
        biomarker_results = {
            'analysis_type': 'aging_biomarkers',
            'timestamp': pd.Timestamp.now().isoformat(),
            'note': 'Placeholder - implement specific biomarker analysis'
        }

        import json
        with open(output_dir / 'biomarker_analysis.json', 'w') as f:
            json.dump(biomarker_results, f, indent=2)

    except Exception as e:
        logger.error(f"Biomarker analysis failed: {e}")
