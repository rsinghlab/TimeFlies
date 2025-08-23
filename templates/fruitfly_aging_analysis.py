"""
TimeFlies Aging Analysis

Project-specific analysis for fruitfly_aging project.
Specialized analysis methods for studying aging in Drosophila melanogaster.

Usage:
- Automatically used when project = "fruitfly_aging" 
- Or explicitly: timeflies analyze --analysis-script templates/fruitfly_aging_analysis.py
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import stats

logger = logging.getLogger(__name__)


def run_analysis(model, config, path_manager, pipeline):
    """
    Main analysis function for aging research.
    
    This function will be called by TimeFlies with the pipeline context.
    """
    logger.info("Starting Aging-Specific Analysis...")
    
    try:
        print("\n🧬 Fruit Fly Aging Analysis")
        print("=" * 50)
        
        # Get experiment directory
        experiment_dir = path_manager.get_experiment_dir(getattr(pipeline, 'experiment_name', None))
        
        # Create analysis directory
        analysis_dir = Path(experiment_dir) / "aging_analysis"
        analysis_dir.mkdir(exist_ok=True)
        
        # Initialize aging analyzer
        analyzer = AgingAnalyzer(config)
        
        # Load predictions if available
        predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
        if predictions_file.exists():
            predictions_df = pd.read_csv(predictions_file)
            print(f"📊 Loaded {len(predictions_df)} predictions")
            analyzer.analyze_predictions(predictions_df, analysis_dir)
        else:
            print("📊 No predictions found - analyzing raw data only")
        
        # Access raw data if available
        if hasattr(pipeline, 'adata_eval') and pipeline.adata_eval is not None:
            adata = pipeline.adata_eval
            print(f"🧬 Analyzing raw data: {adata.n_obs} cells, {adata.n_vars} genes")
            analyzer.analyze_raw_data(adata, analysis_dir)
        elif hasattr(pipeline, 'adata') and pipeline.adata is not None:
            adata = pipeline.adata
            print(f"🧬 Analyzing raw data: {adata.n_obs} cells, {adata.n_vars} genes")
            analyzer.analyze_raw_data(adata, analysis_dir)
        else:
            print("🧬 No raw data available for analysis")
        
        # Generate summary
        analyzer.generate_summary_report(analysis_dir)
        
        print("✅ Aging analysis completed!")
        print(f"📁 Results saved in: {analysis_dir}")
        
    except Exception as e:
        logger.error(f"Aging analysis failed: {e}")
        raise


class AgingAnalyzer:
    """
    Analyzer specialized for aging research in fruit flies.
    """

    def __init__(self, config=None):
        """Initialize aging analyzer."""
        self.config = config
        self.age_groups = self._define_age_groups()
        self.aging_markers = self._load_aging_markers()
        self.results = {}

    def _define_age_groups(self) -> Dict[str, List[int]]:
        """Define age groups based on typical aging data."""
        return {
            "young": [5, 10, 15], 
            "middle": [20, 25, 30, 35], 
            "old": [40, 45, 50], 
            "very_old": [55, 60, 65, 70]
        }

    def _load_aging_markers(self) -> List[str]:
        """Load known aging marker genes for Drosophila."""
        return [
            "InR",    # Insulin receptor
            "foxo",   # Forkhead box O
            "sod1",   # Superoxide dismutase 1
            "cat",    # Catalase
            "hsp70",  # Heat shock protein 70
            "tor",    # Target of rapamycin
            "sir2",   # Sirtuin 2
            "p53",    # Tumor suppressor p53
            "dFOXO",  # Drosophila FOXO
            "Thor",   # 4E-BP (eIF4E-binding protein)
        ]
    
    def analyze_predictions(self, predictions_df, output_dir):
        """Analyze aging predictions."""
        print("\n1️⃣ Prediction Analysis")
        
        try:
            # Basic prediction metrics
            if 'actual_age' in predictions_df.columns and 'predicted_age' in predictions_df.columns:
                mae = np.mean(np.abs(predictions_df['actual_age'] - predictions_df['predicted_age']))
                rmse = np.sqrt(np.mean((predictions_df['actual_age'] - predictions_df['predicted_age'])**2))
                r2 = stats.pearsonr(predictions_df['actual_age'], predictions_df['predicted_age'])[0]**2
                
                print(f"   MAE: {mae:.2f} days")
                print(f"   RMSE: {rmse:.2f} days")
                print(f"   R²: {r2:.3f}")
                
                self.results['prediction_metrics'] = {
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'r2': float(r2)
                }
                
                # Age group performance
                self._analyze_age_group_performance(predictions_df, output_dir)
                
                # Create prediction visualizations
                self._create_prediction_plots(predictions_df, output_dir)
            
            # Sex-specific analysis if available
            if 'sex' in predictions_df.columns:
                self._analyze_sex_differences(predictions_df, output_dir)
                
        except Exception as e:
            logger.error(f"Prediction analysis failed: {e}")
    
    def analyze_raw_data(self, adata, output_dir):
        """Analyze raw single-cell data for aging patterns."""
        print("\n2️⃣ Raw Data Analysis")
        
        try:
            # Check for age information
            age_column = None
            for col in ['age', 'Age', 'AGE', 'age_days']:
                if col in adata.obs.columns:
                    age_column = col
                    break
            
            if age_column is None:
                print("   ⚠️ No age information found in data")
                return
            
            # Age progression analysis
            print(f"   Analyzing age progression using column: {age_column}")
            age_results = self.analyze_age_progression(adata, age_column)
            self.results['age_progression'] = age_results
            
            # Sex-specific aging if available
            sex_column = None
            for col in ['sex', 'Sex', 'SEX', 'gender']:
                if col in adata.obs.columns:
                    sex_column = col
                    break
            
            if sex_column is not None:
                print(f"   Analyzing sex-specific aging using column: {sex_column}")
                sex_results = self.analyze_sex_specific_aging(adata, age_column, sex_column)
                self.results['sex_specific_aging'] = sex_results
            
            # Aging trajectories
            print("   Identifying aging gene trajectories")
            trajectory_results = self.identify_aging_trajectories(adata, age_column)
            self.results['trajectories'] = trajectory_results
            
            # Create raw data visualizations
            self._create_raw_data_plots(adata, age_column, output_dir)
            
        except Exception as e:
            logger.error(f"Raw data analysis failed: {e}")

    def analyze_age_progression(self, adata, age_column: str = "age") -> Dict:
        """Analyze gene expression changes across age groups."""
        results = {}
        ages = sorted(adata.obs[age_column].unique())
        
        print(f"   Found ages: {ages}")
        
        # Correlation with age
        age_correlations = self._compute_age_correlations(adata, age_column)
        results["age_correlations"] = age_correlations
        
        # Top age-correlated genes
        if age_correlations:
            sorted_genes = sorted(age_correlations.items(), 
                                key=lambda x: abs(x[1]['correlation']), reverse=True)
            top_aging_genes = sorted_genes[:20]  # Top 20
            
            print(f"   Top age-correlated genes:")
            for gene, data in top_aging_genes[:5]:  # Print top 5
                corr = data['correlation']
                p_val = data['p_value']
                print(f"     {gene}: r={corr:.3f}, p={p_val:.2e}")
            
            results["top_aging_genes"] = top_aging_genes
        
        # Aging marker expression
        marker_expression = self._analyze_aging_markers(adata, age_column)
        results["aging_markers"] = marker_expression
        
        return results

    def analyze_sex_specific_aging(self, adata, age_column: str = "age", sex_column: str = "sex") -> Dict:
        """Analyze sex-specific patterns in aging."""
        results = {}
        
        # Get unique sexes
        sexes = adata.obs[sex_column].unique()
        print(f"   Found sexes: {list(sexes)}")
        
        # Analyze each sex separately
        for sex in sexes:
            sex_data = adata[adata.obs[sex_column] == sex]
            if len(sex_data) > 10:  # Minimum sample size
                sex_results = self.analyze_age_progression(sex_data, age_column)
                results[f"{sex}_aging"] = sex_results
                print(f"     {sex}: {len(sex_data)} cells analyzed")
        
        return results

    def identify_aging_trajectories(self, adata, age_column: str = "age", method: str = "linear") -> Dict:
        """Identify genes with significant aging trajectories."""
        ages = adata.obs[age_column].values
        trajectories = {}
        significant_count = 0
        
        print(f"   Analyzing {adata.n_vars} genes for aging trajectories...")
        
        for gene_idx, gene in enumerate(adata.var.index):
            try:
                expression = (
                    adata.X[:, gene_idx].toarray().flatten()
                    if hasattr(adata.X, "toarray")
                    else adata.X[:, gene_idx].flatten()
                )
                
                if method == "linear":
                    correlation, p_value = stats.pearsonr(ages, expression)
                    trajectories[gene] = {
                        "correlation": float(correlation),
                        "p_value": float(p_value),
                        "trajectory_type": "increasing" if correlation > 0 else "decreasing",
                    }
                    if p_value < 0.05:
                        significant_count += 1
                        
            except Exception as e:
                continue  # Skip genes with issues
        
        # Filter significant trajectories
        significant_genes = {
            k: v for k, v in trajectories.items() if v["p_value"] < 0.05
        }
        
        print(f"   Found {significant_count} genes with significant aging trajectories")
        
        return {
            "all_trajectories": trajectories,
            "significant_trajectories": significant_genes,
            "method": method,
            "n_significant": significant_count
        }

    def _compute_age_correlations(self, adata, age_column: str) -> Dict:
        """Compute correlations between gene expression and age."""
        ages = adata.obs[age_column].values
        correlations = {}
        
        for gene_idx, gene in enumerate(adata.var.index):
            try:
                expression = (
                    adata.X[:, gene_idx].toarray().flatten()
                    if hasattr(adata.X, "toarray")
                    else adata.X[:, gene_idx].flatten()
                )
                correlation, p_value = stats.pearsonr(ages, expression)
                correlations[gene] = {
                    "correlation": float(correlation), 
                    "p_value": float(p_value)
                }
            except:
                continue  # Skip genes with issues
        
        return correlations

    def _analyze_aging_markers(self, adata, age_column: str) -> Dict:
        """Analyze expression of known aging marker genes."""
        marker_results = {}
        found_markers = []
        
        for marker in self.aging_markers:
            if marker in adata.var.index:
                found_markers.append(marker)
                marker_idx = list(adata.var.index).index(marker)
                expression = (
                    adata.X[:, marker_idx].toarray().flatten()
                    if hasattr(adata.X, "toarray")
                    else adata.X[:, marker_idx].flatten()
                )
                
                # Expression by age
                age_expression = {}
                for age in sorted(adata.obs[age_column].unique()):
                    age_cells = adata.obs[age_column] == age
                    if np.sum(age_cells) > 0:
                        age_expression[str(age)] = {
                            "mean": float(np.mean(expression[age_cells])),
                            "std": float(np.std(expression[age_cells])),
                            "n_cells": int(np.sum(age_cells)),
                        }
                
                marker_results[marker] = age_expression
        
        print(f"   Found {len(found_markers)} aging markers: {found_markers}")
        return marker_results

    def _analyze_age_group_performance(self, predictions_df, output_dir):
        """Analyze prediction performance across age groups."""
        try:
            # Create age quartiles
            age_bins = pd.qcut(predictions_df['actual_age'], q=4, labels=['Young', 'Middle-Young', 'Middle-Old', 'Old'])
            predictions_df['age_group'] = age_bins
            
            group_stats = []
            for group in age_bins.categories:
                group_data = predictions_df[predictions_df['age_group'] == group]
                if len(group_data) > 0:
                    mae = np.mean(np.abs(group_data['actual_age'] - group_data['predicted_age']))
                    rmse = np.sqrt(np.mean((group_data['actual_age'] - group_data['predicted_age'])**2))
                    
                    group_stats.append({
                        'age_group': group,
                        'n_samples': len(group_data),
                        'mae': mae,
                        'rmse': rmse
                    })
            
            group_df = pd.DataFrame(group_stats)
            group_df.to_csv(output_dir / 'age_group_performance.csv', index=False)
            print(f"   Age group performance saved")
            
        except Exception as e:
            logger.error(f"Age group analysis failed: {e}")

    def _analyze_sex_differences(self, predictions_df, output_dir):
        """Analyze sex-specific prediction performance."""
        try:
            sex_stats = []
            for sex in predictions_df['sex'].unique():
                sex_data = predictions_df[predictions_df['sex'] == sex]
                if len(sex_data) > 0:
                    mae = np.mean(np.abs(sex_data['actual_age'] - sex_data['predicted_age']))
                    rmse = np.sqrt(np.mean((sex_data['actual_age'] - sex_data['predicted_age'])**2))
                    
                    sex_stats.append({
                        'sex': sex,
                        'n_samples': len(sex_data),
                        'mae': mae,
                        'rmse': rmse
                    })
            
            sex_df = pd.DataFrame(sex_stats)
            sex_df.to_csv(output_dir / 'sex_performance.csv', index=False)
            print(f"   Sex-specific performance saved")
            
        except Exception as e:
            logger.error(f"Sex analysis failed: {e}")

    def _create_prediction_plots(self, predictions_df, output_dir):
        """Create prediction analysis plots."""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Actual vs Predicted
            axes[0,0].scatter(predictions_df['actual_age'], predictions_df['predicted_age'], alpha=0.6)
            min_age, max_age = predictions_df['actual_age'].min(), predictions_df['actual_age'].max()
            axes[0,0].plot([min_age, max_age], [min_age, max_age], 'r--', alpha=0.5)
            axes[0,0].set_xlabel('Actual Age')
            axes[0,0].set_ylabel('Predicted Age')
            axes[0,0].set_title('Age Prediction Accuracy')
            
            # Residuals
            residuals = predictions_df['predicted_age'] - predictions_df['actual_age']
            axes[0,1].scatter(predictions_df['actual_age'], residuals, alpha=0.6)
            axes[0,1].axhline(0, color='red', linestyle='--', alpha=0.5)
            axes[0,1].set_xlabel('Actual Age')
            axes[0,1].set_ylabel('Residuals (Predicted - Actual)')
            axes[0,1].set_title('Prediction Residuals')
            
            # Age distribution
            axes[1,0].hist(predictions_df['actual_age'], bins=20, alpha=0.7, color='skyblue')
            axes[1,0].set_xlabel('Age')
            axes[1,0].set_ylabel('Count')
            axes[1,0].set_title('Age Distribution')
            
            # Error distribution
            errors = np.abs(predictions_df['actual_age'] - predictions_df['predicted_age'])
            axes[1,1].hist(errors, bins=20, alpha=0.7, color='lightcoral')
            axes[1,1].set_xlabel('Absolute Error')
            axes[1,1].set_ylabel('Count')
            axes[1,1].set_title('Prediction Error Distribution')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'aging_prediction_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Prediction plots failed: {e}")

    def _create_raw_data_plots(self, adata, age_column, output_dir):
        """Create raw data analysis plots."""
        try:
            # Simple aging marker expression plot if markers are found
            marker_data = []
            marker_names = []
            ages = []
            
            for marker in self.aging_markers:
                if marker in adata.var.index:
                    marker_idx = list(adata.var.index).index(marker)
                    expression = (
                        adata.X[:, marker_idx].toarray().flatten()
                        if hasattr(adata.X, "toarray")
                        else adata.X[:, marker_idx].flatten()
                    )
                    marker_data.extend(expression)
                    marker_names.extend([marker] * len(expression))
                    ages.extend(adata.obs[age_column].values)
            
            if marker_data:
                marker_df = pd.DataFrame({
                    'expression': marker_data,
                    'marker': marker_names,
                    'age': ages
                })
                
                plt.figure(figsize=(12, 8))
                for marker in marker_df['marker'].unique():
                    marker_subset = marker_df[marker_df['marker'] == marker]
                    age_means = marker_subset.groupby('age')['expression'].mean()
                    plt.plot(age_means.index, age_means.values, 'o-', label=marker, alpha=0.7)
                
                plt.xlabel('Age')
                plt.ylabel('Expression')
                plt.title('Aging Marker Expression by Age')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(output_dir / 'aging_markers_expression.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except Exception as e:
            logger.error(f"Raw data plots failed: {e}")

    def generate_summary_report(self, output_dir):
        """Generate comprehensive summary report."""
        print("\n3️⃣ Generating Summary Report")
        
        try:
            summary = {
                'analysis_type': 'aging_analysis',
                'timestamp': pd.Timestamp.now().isoformat(),
                'results': self.results,
                'aging_markers_analyzed': self.aging_markers,
                'age_groups': self.age_groups
            }
            
            # Save summary
            import json
            with open(output_dir / 'aging_analysis_summary.json', 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"   Summary saved: {output_dir / 'aging_analysis_summary.json'}")
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")


# For backward compatibility
def main():
    """Legacy main function."""
    print("This script should now be used via: timeflies analyze --analysis-script templates/fruitfly_aging_analysis.py")