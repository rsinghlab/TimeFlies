"""
TimeFlies Custom Analysis Script Template

This template shows how to create custom analysis scripts that can be run with:
timeflies analyze --analysis-script /path/to/your_analysis.py

Required function:
- run_analysis(model, config, path_manager, pipeline)

Available objects:
- model: Trained model instance (if available)
- config: Full configuration object with all settings
- path_manager: PathManager for handling file paths
- pipeline: Full PipelineManager instance for accessing all methods

Example usage:
timeflies analyze --analysis-script templates/custom_analysis_example.py
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


def run_analysis(model, config, path_manager, pipeline):
    """
    Main analysis function - REQUIRED.
    
    This function will be called by TimeFlies with the pipeline context.
    
    Args:
        model: Trained model instance (may be None if no model loaded)
        config: Configuration object with project settings
        path_manager: PathManager for file operations
        pipeline: Complete PipelineManager instance
    """
    logger.info("Starting custom analysis...")
    
    try:
        # Get project information
        project = getattr(config, 'project', 'unknown')
        tissue = config.data.tissue
        model_type = config.data.model
        
        print(f"Running custom analysis for:")
        print(f"  Project: {project}")
        print(f"  Tissue: {tissue}")
        print(f"  Model: {model_type}")
        
        # Example 1: Access predictions if they exist
        try:
            experiment_dir = path_manager.get_experiment_dir(getattr(pipeline, 'experiment_name', None))
            predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
            
            if predictions_file.exists():
                predictions_df = pd.read_csv(predictions_file)
                print(f"  Loaded predictions: {len(predictions_df)} samples")
                
                # Example analysis: Age distribution
                if 'true_age' in predictions_df.columns and 'predicted_age' in predictions_df.columns:
                    mae = np.mean(np.abs(predictions_df['true_age'] - predictions_df['predicted_age']))
                    print(f"  Mean Absolute Error: {mae:.2f}")
                    
                    # Age group analysis
                    age_groups = pd.cut(predictions_df['true_age'], bins=3, labels=['Young', 'Middle', 'Old'])
                    group_errors = predictions_df.groupby(age_groups).apply(
                        lambda x: np.mean(np.abs(x['true_age'] - x['predicted_age']))
                    )
                    print("  Age Group Errors:")
                    for group, error in group_errors.items():
                        print(f"    {group}: {error:.2f}")
            else:
                print("  No predictions file found - run evaluation first")
                
        except Exception as e:
            logger.warning(f"Could not load predictions: {e}")
        
        # Example 2: Custom model analysis (if model available)
        if model is not None:
            try:
                print(f"  Model type: {type(model).__name__}")
                if hasattr(model, 'summary'):
                    print("  Model summary available")
                if hasattr(model, 'get_weights'):
                    weights = model.get_weights()
                    print(f"  Model has {len(weights)} weight arrays")
            except Exception as e:
                logger.warning(f"Model analysis failed: {e}")
        
        # Example 3: Create custom output files
        try:
            # Create custom analysis directory
            analysis_dir = Path(experiment_dir) / "custom_analysis"
            analysis_dir.mkdir(exist_ok=True)
            
            # Save custom results
            results = {
                'project': project,
                'tissue': tissue,
                'model': model_type,
                'analysis_timestamp': pd.Timestamp.now().isoformat(),
                'custom_metric': np.random.rand()  # Replace with your analysis
            }
            
            results_file = analysis_dir / "custom_results.json"
            import json
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"  Custom results saved to: {results_file}")
            
        except Exception as e:
            logger.warning(f"Could not save custom results: {e}")
        
        # Example 4: Access raw data (if needed)
        try:
            if hasattr(pipeline, 'adata_eval') and pipeline.adata_eval is not None:
                print(f"  Evaluation data: {pipeline.adata_eval.n_obs} cells, {pipeline.adata_eval.n_vars} genes")
            elif hasattr(pipeline, 'adata') and pipeline.adata is not None:
                print(f"  Full data: {pipeline.adata.n_obs} cells, {pipeline.adata.n_vars} genes")
        except Exception as e:
            logger.warning(f"Data access failed: {e}")
        
        print("✅ Custom analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Custom analysis failed: {e}")
        raise


# Optional: Additional helper functions
def calculate_age_acceleration(predictions_df):
    """Example helper function for age acceleration analysis."""
    if 'true_age' in predictions_df.columns and 'predicted_age' in predictions_df.columns:
        return predictions_df['predicted_age'] - predictions_df['true_age']
    return None


def analyze_gene_importance(model, feature_names):
    """Example helper function for feature importance analysis."""
    try:
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'gene': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            return importance_df
    except:
        pass
    return None