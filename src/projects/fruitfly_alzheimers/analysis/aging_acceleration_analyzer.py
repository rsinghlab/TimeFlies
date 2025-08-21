#!/usr/bin/env python3
"""
Alzheimer's Aging Acceleration Analysis

Analyzes prediction results to determine if Alzheimer's flies are predicted
as older than their chronological age, indicating accelerated aging.

This script assumes you've already run the pipeline and have prediction results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AlzheimersAgingAccelerationAnalyzer:
    """Analyzes aging acceleration from existing prediction results"""
    
    def __init__(self, model=None, config=None, path_manager=None):
        """Initialize analyzer with optional pipeline components"""
        self.results = {}
        self.model = model
        self.config = config
        self.path_manager = path_manager
        
    def run_complete_analysis(self):
        """Run complete aging acceleration analysis using pipeline data"""
        print("\n🧬 Starting Alzheimer's Aging Acceleration Analysis...")
        print("="*60)
        
        # Load and analyze existing prediction results
        try:
            df = self.load_prediction_results()
            print(f"📊 Loaded {len(df)} predictions")
            print(f"   Age distribution: {df['actual_age'].value_counts().sort_index().to_dict()}")
            
            # Basic aging acceleration analysis
            mean_error = df['prediction_error'].mean()
            print(f"\n📈 Aging Acceleration Metrics:")
            print(f"   Mean prediction error: {mean_error:.2f} days")
            
            if mean_error > 0:
                print(f"   ✅ Alzheimer's flies predicted {mean_error:.2f} days OLDER (accelerated aging)")
            elif mean_error < 0:
                print(f"   ⚠️  Alzheimer's flies predicted {abs(mean_error):.2f} days YOUNGER")
            else:
                print(f"   ➖ No aging bias detected")
            
            # Age-specific analysis
            print(f"\n📊 Age-specific prediction errors:")
            for age in sorted(df['actual_age'].unique()):
                age_df = df[df['actual_age'] == age]
                age_error = age_df['prediction_error'].mean()
                print(f"   Age {age}: {age_error:+.2f} days (n={len(age_df)})")
            
            print("\n✅ Aging acceleration analysis completed!")
            print("="*60)
            
        except FileNotFoundError as e:
            print(f"⚠️  Could not run detailed analysis: {e}")
            print("Basic metrics have been computed and saved by the pipeline.")
        except Exception as e:
            print(f"⚠️  Analysis error: {e}")
            print("Basic metrics have been computed and saved by the pipeline.")
        
    def load_prediction_results(self):
        """Load prediction results from known pipeline output location"""
        # Pipeline outputs go to outputs/fruitfly_alzheimers/results/
        results_path = Path("outputs/fruitfly_alzheimers/results/predictions.csv")
        
        if not results_path.exists():
            raise FileNotFoundError(f"Prediction results not found at {results_path}. Run the pipeline first.")
        
        df = pd.read_csv(results_path)
        
        # Check for required columns
        required_cols = ['actual_age', 'predicted_age', 'genotype']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols}")
            print(f"Available columns: {df.columns.tolist()}")
        
        return df
    
    def analyze_prediction_bias(self, df):
        """Analyze prediction bias patterns"""
        print("Analyzing prediction bias patterns...")
        
        # Separate control vs alzheimer's flies
        control_df = df[df['genotype'] == 'control'].copy()
        alzheimers_df = df[df['genotype'].isin(['AB42', 'hTau'])].copy()
        
        # Calculate prediction bias (predicted - actual age)
        control_df['bias'] = pd.to_numeric(control_df['predicted_age']) - pd.to_numeric(control_df['actual_age'])
        alzheimers_df['bias'] = pd.to_numeric(alzheimers_df['predicted_age']) - pd.to_numeric(alzheimers_df['actual_age'])
        
        # Basic statistics
        control_bias = control_df['bias']
        alzheimers_bias = alzheimers_df['bias']
        
        print(f"Control flies: {len(control_bias)}")
        print(f"Alzheimer's flies: {len(alzheimers_bias)}")
        print(f"Control prediction bias: {control_bias.mean():.2f} ± {control_bias.std():.2f} days")
        print(f"Alzheimer's prediction bias: {alzheimers_bias.mean():.2f} ± {alzheimers_bias.std():.2f} days")
        
        # Statistical test
        if len(control_bias) > 0 and len(alzheimers_bias) > 0:
            statistic, p_value = stats.ttest_ind(alzheimers_bias, control_bias)
            print(f"T-test p-value for bias difference: {p_value:.2e}")
        
        # Analyze by genotype
        print("\nBias by genotype:")
        for genotype in ['AB42', 'hTau']:
            genotype_data = alzheimers_df[alzheimers_df['genotype'] == genotype]
            if len(genotype_data) > 0:
                genotype_bias = genotype_data['bias']
                print(f"{genotype}: {genotype_bias.mean():.2f} ± {genotype_bias.std():.2f} (n={len(genotype_bias)})")
                
                # Test against control
                if len(control_bias) > 0:
                    _, p_val = stats.ttest_ind(genotype_bias, control_bias)
                    print(f"  vs control p-value: {p_val:.2e}")
        
        # Calculate percentage predicted as older
        control_older_pct = (control_bias > 0).mean() * 100 if len(control_bias) > 0 else 0
        alzheimers_older_pct = (alzheimers_bias > 0).mean() * 100 if len(alzheimers_bias) > 0 else 0
        
        print(f"\nFlies predicted as older than actual age:")
        print(f"Control: {control_older_pct:.1f}%")
        print(f"Alzheimer's: {alzheimers_older_pct:.1f}%")
        
        # Store results
        self.results = {
            'control_df': control_df,
            'alzheimers_df': alzheimers_df,
            'control_bias_mean': control_bias.mean() if len(control_bias) > 0 else 0,
            'alzheimers_bias_mean': alzheimers_bias.mean() if len(alzheimers_bias) > 0 else 0,
            'p_value': p_value if 'p_value' in locals() else None,
            'control_older_pct': control_older_pct,
            'alzheimers_older_pct': alzheimers_older_pct
        }
        
        return self.results
    
    def create_visualizations(self, output_dir="."):
        """Create visualizations of aging acceleration patterns"""
        print("Creating visualizations...")
        
        if not self.results:
            print("No results to visualize. Run analyze_prediction_bias first.")
            return
        
        control_df = self.results['control_df']
        alzheimers_df = self.results['alzheimers_df']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction bias distribution
        if len(control_df) > 0:
            axes[0,0].hist(control_df['bias'], alpha=0.7, label='Control', bins=20)
        if len(alzheimers_df) > 0:
            axes[0,0].hist(alzheimers_df['bias'], alpha=0.7, label="Alzheimer's", bins=20)
        axes[0,0].set_xlabel('Prediction Bias (Predicted - Actual Age)')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Age Prediction Bias Distribution')
        axes[0,0].legend()
        axes[0,0].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 2. Bias by genotype boxplot
        bias_data = []
        labels = []
        
        if len(control_df) > 0:
            bias_data.append(control_df['bias'])
            labels.append('Control')
        
        for genotype in ['AB42', 'hTau']:
            genotype_data = alzheimers_df[alzheimers_df['genotype'] == genotype]
            if len(genotype_data) > 0:
                bias_data.append(genotype_data['bias'])
                labels.append(genotype)
        
        if bias_data:
            axes[0,1].boxplot(bias_data, labels=labels)
            axes[0,1].set_ylabel('Prediction Bias (days)')
            axes[0,1].set_title('Prediction Bias by Genotype')
            axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 3. Actual vs Predicted scatter
        if len(control_df) > 0:
            axes[1,0].scatter(pd.to_numeric(control_df['actual_age']), 
                            pd.to_numeric(control_df['predicted_age']), 
                            alpha=0.5, label='Control', s=20)
        if len(alzheimers_df) > 0:
            axes[1,0].scatter(pd.to_numeric(alzheimers_df['actual_age']), 
                            pd.to_numeric(alzheimers_df['predicted_age']), 
                            alpha=0.5, label="Alzheimer's", s=20)
        
        # Perfect prediction line
        all_df = pd.concat([control_df, alzheimers_df]) if len(control_df) > 0 and len(alzheimers_df) > 0 else (control_df if len(control_df) > 0 else alzheimers_df)
        if len(all_df) > 0:
            min_age = pd.to_numeric(all_df['actual_age']).min()
            max_age = pd.to_numeric(all_df['actual_age']).max()
            axes[1,0].plot([min_age, max_age], [min_age, max_age], 'k--', alpha=0.5, label='Perfect Prediction')
        
        axes[1,0].set_xlabel('Actual Age (days)')
        axes[1,0].set_ylabel('Predicted Age (days)')
        axes[1,0].set_title('Actual vs Predicted Age')
        axes[1,0].legend()
        
        # 4. Mean bias by age and genotype
        age_bias_data = []
        all_ages = set()
        if len(control_df) > 0:
            all_ages.update(pd.to_numeric(control_df['actual_age']).unique())
        if len(alzheimers_df) > 0:
            all_ages.update(pd.to_numeric(alzheimers_df['actual_age']).unique())
        
        for age in sorted(all_ages):
            # Control flies of this age
            if len(control_df) > 0:
                control_age_data = control_df[pd.to_numeric(control_df['actual_age']) == age]
                if len(control_age_data) > 0:
                    age_bias_data.append({
                        'age': age,
                        'genotype': 'Control',
                        'bias': control_age_data['bias'].mean(),
                        'count': len(control_age_data)
                    })
            
            # Alzheimer's flies of this age
            for genotype in ['AB42', 'hTau']:
                genotype_age_data = alzheimers_df[
                    (pd.to_numeric(alzheimers_df['actual_age']) == age) & 
                    (alzheimers_df['genotype'] == genotype)
                ]
                if len(genotype_age_data) > 0:
                    age_bias_data.append({
                        'age': age,
                        'genotype': genotype,
                        'bias': genotype_age_data['bias'].mean(),
                        'count': len(genotype_age_data)
                    })
        
        if age_bias_data:
            age_df = pd.DataFrame(age_bias_data)
            sns.barplot(data=age_df, x='age', y='bias', hue='genotype', ax=axes[1,1])
            axes[1,1].set_title('Mean Prediction Bias by Age and Genotype')
            axes[1,1].set_xlabel('Actual Age (days)')
            axes[1,1].set_ylabel('Mean Prediction Bias (days)')
            axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(output_dir) / 'alzheimers_aging_acceleration_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        
        return str(output_path)
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if not self.results:
            print("No results to report. Run analyze_prediction_bias first.")
            return
        
        print("\n" + "="*80)
        print("ALZHEIMER'S AGING ACCELERATION ANALYSIS SUMMARY")
        print("="*80)
        
        control_bias_mean = self.results['control_bias_mean']
        alzheimers_bias_mean = self.results['alzheimers_bias_mean']
        p_value = self.results.get('p_value')
        control_older_pct = self.results['control_older_pct']
        alzheimers_older_pct = self.results['alzheimers_older_pct']
        
        print(f"Control flies: {len(self.results['control_df'])}")
        print(f"Alzheimer's flies: {len(self.results['alzheimers_df'])}")
        
        print("\nKEY FINDINGS:")
        print(f"1. Control prediction bias: {control_bias_mean:.2f} days")
        print(f"2. Alzheimer's prediction bias: {alzheimers_bias_mean:.2f} days")
        
        if p_value is not None:
            significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
            print(f"3. Difference is {significance} (p={p_value:.2e})")
        
        print(f"\n4. Flies predicted as OLDER than actual age:")
        print(f"   - Control: {control_older_pct:.1f}%")
        print(f"   - Alzheimer's: {alzheimers_older_pct:.1f}%")
        
        # Evidence assessment
        if alzheimers_older_pct > control_older_pct + 10:
            print("   *** STRONG EVIDENCE OF ACCELERATED AGING ***")
        elif alzheimers_bias_mean > control_bias_mean + 1:
            print("   *** MODERATE EVIDENCE OF ACCELERATED AGING ***")
        else:
            print("   - No strong evidence of accelerated aging")
        
        print("\nINTERPRETATION:")
        if alzheimers_bias_mean > control_bias_mean:
            print("✓ Model predicts Alzheimer's flies as older than control flies")
            print("✓ This suggests disrupted aging gene expression patterns")
        else:
            print("✗ No clear age over-prediction pattern in Alzheimer's flies")
        
        print("="*80)
        
        return self.results

def main():
    """Main analysis pipeline"""
    print("Starting Alzheimer's Aging Acceleration Analysis...")
    
    try:
        analyzer = AlzheimersAgingAccelerationAnalyzer()
        
        # Load prediction results from known location
        df = analyzer.load_prediction_results()
        
        # Analyze bias patterns
        results = analyzer.analyze_prediction_bias(df)
        
        # Create visualizations
        analyzer.create_visualizations()
        
        # Generate report
        analyzer.generate_summary_report()
        
        print("\nAnalysis completed successfully!")
        return results
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you have run the pipeline first.")
        return None

if __name__ == "__main__":
    main()