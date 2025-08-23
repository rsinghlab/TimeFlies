# eda_handler.py

import pandas as pd
import numpy as np
from scipy import stats
import scipy
import json
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from .visuals import VisualizationTools


class EDAHandler:
    """
    A class to handle Exploratory Data Analysis (EDA) on multiple datasets.

    This class takes in multiple datasets and performs EDA on them, leveraging
    configuration settings to customize the analysis.
    """

    def __init__(
        self,
        config,
        path_manager=None,
        adata=None,
        adata_eval=None,
        adata_original=None,
        adata_corrected=None,
        adata_eval_corrected=None,
        output_dir=None,
    ):
        """
        Initializes the EDAHandler with the given configuration and datasets.

        Parameters:
        - config (ConfigHandler): Configuration settings for the EDA.
        - path_manager (PathManager): Path manager object to handle directory paths.
        - adata (AnnData): Main AnnData object containing the dataset.
        - adata_eval (AnnData): AnnData object for evaluation data.
        - adata_original (AnnData): The original unfiltered AnnData object.
        - adata_corrected (AnnData): The batch-corrected AnnData object.
        - adata_eval_corrected (AnnData): The batch-corrected evaluation AnnData object.
        """
        self.config = config
        self.path_manager = path_manager
        self.adata = adata
        self.adata_eval = adata_eval
        self.adata_original = adata_original
        self.adata_corrected = adata_corrected
        self.adata_eval_corrected = adata_eval_corrected
        self.output_dir = output_dir or "outputs/eda"
        
        # Store results for report generation
        self.eda_results = {
            "summary": {},
            "visualizations": [],
            "statistics": {},
            "tables": {}
        }

        # Initialize VisualizationTools with config and path_manager
        if path_manager:
            self.visual_tools = VisualizationTools(
                config=self.config, path_manager=self.path_manager
            )
        else:
            self.visual_tools = VisualizationTools(
                config=self.config, path_manager=None, output_dir=self.output_dir
            )

    def _print_general_info(self, adata):
        """
        Print general information about the dataset including its shape,
        variable names, and columns in 'obs'.

        Args:
            adata (AnnData): The input dataset.
        """
        print(f"\nData Shape: {adata.shape}")
        print(f"\nVariable names: {adata.var_names}")
        print(f"\nColumns in 'obs': {adata.obs.columns}")

    def _display_class_info(self, adata, encoding_column, folder_name, dataset_name):
        """
        Display information about the unique classes and class counts in the dataset.

        Args:
            adata (AnnData): The input dataset.
            encoding_column (str): Column used for encoding class labels.
            folder_name (str): Name of the folder where visualizations will be saved.
            dataset_name (str): Name of the dataset for labeling outputs.
        """
        unique_classes = np.unique(adata.obs[encoding_column])
        class_mapping = {
            class_name: f"Class {idx}" for idx, class_name in enumerate(unique_classes)
        }

        print("\nUnique Classes:")
        for class_name in unique_classes:
            print(f"{class_mapping[class_name]}: {class_name}")

        num_classes = len(unique_classes)
        print(f"\nNumber of unique classes: {num_classes}")

        # Class distribution
        class_counts = adata.obs[encoding_column].value_counts().sort_values()
        print(f"\nClass Counts:\n{class_counts}")

        # Plot class distribution
        self.visual_tools.plot_class_distribution(
            class_counts=class_counts,
            file_name=f"{dataset_name}_class_distribution.png",
            dataset=folder_name,
            subfolder_name="EDA",
        )

    def _analyze_gene_statistics(self, adata, folder_name, dataset_name):
        """
        Perform statistical analysis and outlier detection for gene expression data.

        Args:
            adata (AnnData): The input dataset.
            folder_name (str): Name of the folder where visualizations will be saved.
            dataset_name (str): Name of the dataset for labeling outputs.
        """
        # Statistical summary of the first 10 genes
        data_arr = adata.X.toarray() if scipy.sparse.issparse(adata.X) else adata.X
        stats_df = self._generate_stat_summary(data_arr, adata.var_names)
        print(f"\nStatistical Summary for the first 10 genes:\n{stats_df}")

        # Outlier detection for the first 10 genes
        num_outliers = self._detect_outliers(data_arr, threshold=3)
        print(f"\nNumber of outliers in the gene expression data: {num_outliers}")

        # Visualize gene expression overview
        gene_expression_matrix = data_arr[:5, :5]
        gene_expression_df = pd.DataFrame(
            data=gene_expression_matrix,
            index=adata.obs.index[:5],
            columns=adata.var.index[:5],
        )
        self.visual_tools.create_styled_dataframe(
            df=gene_expression_df,
            subfolder_name="EDA",
            dataset=folder_name,
            file_name=f"{dataset_name}_sampled_gene_expression_overview.png",
        )

        self.visual_tools.create_styled_dataframe(
            df=stats_df,
            subfolder_name="EDA",
            dataset=folder_name,
            file_name=f"{dataset_name}_gene_expression_statistics.png",
        )

    def _handle_missing_values(self, adata):
        """
        Calculate and return the number of missing values in the dataset.

        Args:
            adata (AnnData): Input AnnData object.

        Returns:
            int: Number of missing values in the dataset.
        """
        if scipy.sparse.issparse(adata.X):
            data_arr = adata.X.toarray()
        else:
            data_arr = adata.X
        return np.isnan(data_arr).sum()

    def _check_for_duplicates(self, df):
        """
        Checks for duplicate rows in the given DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            num_duplicates (int): Number of duplicate rows.
            duplicate_rows (pd.DataFrame): DataFrame of duplicate rows.
        """
        duplicate_rows = df.duplicated()
        num_duplicates = duplicate_rows.sum()
        return num_duplicates, df[duplicate_rows]

    def _detect_outliers(self, data, threshold=3):
        """
        Detects outliers in the given data based on the z-score.

        Args:
            data (np.array): Input data.
            threshold (float): The z-score threshold to define outliers.

        Returns:
            int: Number of outliers detected.
        """
        z_scores = np.abs(stats.zscore(data[:, :10], axis=0))
        return np.where(z_scores > threshold)[0].shape[0]

    def _generate_stat_summary(self, data, var_names, num_genes=10):
        """
        Generates a statistical summary for the first few genes.

        Args:
            data (np.array): Gene expression data.
            var_names (pd.Index): Variable names (gene names) corresponding to the data columns.
            num_genes (int): The number of genes to include in the summary (default: 10).

        Returns:
            pd.DataFrame: A DataFrame containing the statistical summary for the specified number of genes.
        """
        # Generate a statistical summary for the first `num_genes` genes
        stats_df = (
            pd.DataFrame(data[:, :num_genes], columns=var_names[:num_genes])
            .describe()
            .transpose()
        )
        return stats_df

    def eda(self, adata, encoding_column, folder_name, dataset_name):
        """
        Perform exploratory data analysis on the dataset.

        Args:
            adata (AnnData): The input data for analysis.
            encoding_column (str): Column used for encoding class labels.
            folder_name (str): Name of the folder where visualizations will be saved.
            dataset_name (str): Name of the dataset for labeling outputs.

        Returns:
            None
        """
        # Display the first 5 rows of the 'obs' dataframe
        print("\nFirst 5 rows of 'obs' dataframe:")
        print(adata.obs.head(5))

        # Visualize the 'obs' dataframe
        self.visual_tools.create_styled_dataframe(
            df=adata.obs.head(5),
            subfolder_name="EDA",
            dataset=folder_name,
            file_name=f"{dataset_name}_observation_df_overview.png",
        )

        # Print and visualize basic dataset information
        self._print_general_info(adata)

        # Visualize sparse vs dense matrices
        self.visual_tools.visualize_sparse_vs_dense(
            adata=adata,
            subset_size=50,
            head_size=5,
            file_name=f"{dataset_name}_sparse_vs_dense.png",
            dataset=folder_name,
            subfolder_name="EDA",
        )

        # Handle duplicate rows using a utility function
        adata.obs = adata.obs.reset_index(drop=False)
        num_duplicates, duplicate_rows = self._check_for_duplicates(adata.obs)
        print(f"\nNumber of duplicate rows in 'obs': {num_duplicates}")
        if num_duplicates > 0:
            print("Duplicate rows in 'obs':")
            print(duplicate_rows)
        adata.obs.set_index("index", inplace=True)

        # Handle missing values using a utility function
        missing_values = self._handle_missing_values(adata)
        print(f"\nMissing Values: {missing_values}")

        # Display unique classes and class counts
        self._display_class_info(
            adata=adata,
            encoding_column=encoding_column,
            folder_name=folder_name,
            dataset_name=dataset_name,
        )

        # Perform statistical analysis and outlier detection
        self._analyze_gene_statistics(
            adata=adata, folder_name=folder_name, dataset_name=dataset_name
        )

    def run_eda(self):
        """
        Runs EDA based on the configuration settings, deciding whether to
        analyze corrected or uncorrected datasets, depending on the batch correction.

        Returns:
            None
        """
        if getattr(self.config.data.batch_correction, "enabled", False):
            # Run EDA on batch-corrected datasets
            self.eda(
                adata=self.adata_corrected,
                encoding_column=getattr(self.config.data, "target_variable", "age"),
                folder_name="Batch Training Data",
                dataset_name="batch_train",
            )
            self.eda(
                adata=self.adata_eval_corrected,
                encoding_column=getattr(self.config.data, "target_variable", "age"),
                folder_name="Batch Evaluation Data",
                dataset_name="batch_evaluation",
            )
        else:
            # Run EDA on uncorrected datasets
            self.eda(
                adata=self.adata,
                encoding_column=getattr(self.config.data, "target_variable", "age"),
                folder_name="Training Data",
                dataset_name="train",
            )
            self.eda(
                adata=self.adata_eval,
                encoding_column=getattr(self.config.data, "target_variable", "age"),
                folder_name="Evaluation Data",
                dataset_name="evaluation",
            )
            self.eda(
                adata=self.adata_original,
                encoding_column=getattr(self.config.data, "target_variable", "age"),
                folder_name="Original Data",
                dataset_name="original",
            )
    
    def run_comprehensive_eda(self, split="all"):
        """
        Run comprehensive EDA on specified data split and save results.
        
        Args:
            split: "all", "train", or "test" - which data to analyze
        """
        # Load data if not already loaded
        if self.adata is None:
            self._load_data()
            
        # Select appropriate dataset based on split
        if split == "train":
            data = self._get_train_subset()
            split_desc = "Training Data"
        elif split == "test":
            data = self._get_test_subset()
            split_desc = "Test Data"
        else:
            data = self.adata if self.adata is not None else self.adata_original
            split_desc = "All Data"
            
        if data is None:
            raise ValueError(f"No data available for split: {split}")
            
        print(f"\n📊 Running comprehensive EDA on {split_desc}")
        print(f"   Samples: {data.n_obs:,}")
        print(f"   Genes: {data.n_vars:,}")
        
        # 1. Basic Statistics
        self._compute_basic_stats(data, split_desc)
        
        # 2. Distribution Analysis
        self._analyze_distributions(data, split_desc)
        
        # 3. Correlation Analysis
        self._analyze_correlations(data, split_desc)
        
        # 4. Dimensionality Reduction (skip for large datasets to avoid hanging)
        if data.n_obs < 50000:  # Only for datasets smaller than 50k cells
            self._perform_dim_reduction(data, split_desc)
        else:
            print(f"   ⏩ Skipping dimensionality reduction for large dataset ({data.n_obs} cells)")
        
        # 5. Gene Expression Analysis
        self._analyze_gene_expression(data, split_desc)
        
        # 6. Save summary
        self._save_eda_summary()
        
        print(f"\n✅ EDA complete! Results saved to: {self.output_dir}")
    
    def _load_data(self):
        """Load data based on configuration."""
        from ..data import DataLoader
        
        print("Loading data for EDA...")
        data_loader = DataLoader(self.config)
        
        if self.config.data.batch_correction.enabled:
            self.adata = data_loader.load_batch_corrected_data("train")
            self.adata_eval = data_loader.load_batch_corrected_data("eval")
            self.adata_original = None  # Original not needed for batch corrected
        else:
            # load_data() returns tuple (train, eval, original)
            self.adata, self.adata_eval, self.adata_original = data_loader.load_data()
    
    def _get_train_subset(self):
        """Get training subset based on config."""
        if self.adata is not None:
            return self.adata
            
        # Load and filter based on config
        data = self.adata_original
        if data is None:
            return None
            
        # Apply split configuration
        train_values = getattr(self.config.data.split, 'train', [])
        if train_values and hasattr(self.config.data.split, 'column'):
            column = self.config.data.split.column
            if column in data.obs.columns:
                mask = data.obs[column].isin(train_values)
                return data[mask].copy()
        
        return data
    
    def _get_test_subset(self):
        """Get test subset based on config."""
        if self.adata_eval is not None:
            return self.adata_eval
            
        # Load and filter based on config
        data = self.adata_original or self.adata_eval
        if data is None:
            return None
            
        # Apply split configuration
        test_values = getattr(self.config.data.split, 'test', [])
        if test_values and hasattr(self.config.data.split, 'column'):
            column = self.config.data.split.column
            if column in data.obs.columns:
                mask = data.obs[column].isin(test_values)
                return data[mask].copy()
        
        return data
    
    def _compute_basic_stats(self, data, split_desc):
        """Compute and save basic statistics."""
        print("   📈 Computing basic statistics...")
        
        stats = {
            "n_samples": data.n_obs,
            "n_genes": data.n_vars,
            "memory_mb": data.X.nbytes / 1024 / 1024 if hasattr(data.X, 'nbytes') else 0,
            "sparsity": 1.0 - (data.X.nnz / np.prod(data.X.shape)) if hasattr(data.X, 'nnz') else 0,
        }
        
        # Age statistics if available
        if 'age' in data.obs.columns:
            age_col = data.obs['age']
            # Handle both numeric and categorical age data
            if pd.api.types.is_numeric_dtype(age_col):
                stats["age_mean"] = float(age_col.mean())
                stats["age_std"] = float(age_col.std())
                stats["age_min"] = float(age_col.min())
                stats["age_max"] = float(age_col.max())
            else:
                # If categorical, just show value counts
                stats["age_values"] = age_col.value_counts().to_dict()
        
        # Cell type statistics
        for col in ['cell_type', 'genotype', 'sex', 'tissue']:
            if col in data.obs.columns:
                value_counts = data.obs[col].value_counts()
                stats[f"{col}_counts"] = value_counts.to_dict()
        
        self.eda_results["summary"] = stats
        
        # Save to JSON
        stats_file = Path(self.output_dir) / "summary_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
    
    def _analyze_distributions(self, data, split_desc):
        """Analyze and plot distributions."""
        print("   📊 Analyzing distributions...")
        
        output_path = Path(self.output_dir)
        
        # Age distribution
        if 'age' in data.obs.columns:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            age_col = data.obs['age']
            if pd.api.types.is_numeric_dtype(age_col):
                # Numeric age - histogram
                axes[0].hist(age_col, bins=30, edgecolor='black', alpha=0.7)
                axes[0].set_xlabel('Age (days)')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Age Distribution')
                
                # Box plot by genotype if available
                if 'genotype' in data.obs.columns:
                    genotypes = data.obs['genotype'].unique()
                    age_by_genotype = [data.obs[data.obs['genotype'] == g]['age'] for g in genotypes]
                    axes[1].boxplot(age_by_genotype, labels=genotypes)
                    axes[1].set_xlabel('Genotype')
                    axes[1].set_ylabel('Age (days)')
                    axes[1].set_title('Age by Genotype')
            else:
                # Categorical age - bar plot
                age_counts = age_col.value_counts()
                axes[0].bar(age_counts.index, age_counts.values, alpha=0.7)
                axes[0].set_xlabel('Age Category')
                axes[0].set_ylabel('Count')
                axes[0].set_title('Age Distribution')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Stacked bar by genotype if available
                if 'genotype' in data.obs.columns:
                    crosstab = pd.crosstab(age_col, data.obs['genotype'])
                    crosstab.plot(kind='bar', ax=axes[1], stacked=True)
                    axes[1].set_xlabel('Age Category')
                    axes[1].set_ylabel('Count')
                    axes[1].set_title('Age by Genotype')
                    axes[1].tick_params(axis='x', rotation=45)
                    axes[1].legend(title='Genotype')
            
            plt.tight_layout()
            fig.savefig(output_path / "age_distribution.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.eda_results["visualizations"].append("age_distribution.png")
    
    def _analyze_correlations(self, data, split_desc):
        """Analyze correlations between features."""
        print("   🔗 Analyzing correlations...")
        
        # Select top variable genes for correlation
        if hasattr(data, 'var'):
            # Calculate variance for each gene
            if scipy.sparse.issparse(data.X):
                gene_vars = np.array(data.X.todense()).var(axis=0)
            else:
                gene_vars = data.X.var(axis=0)
            
            # Select top 50 most variable genes
            top_genes_idx = np.argsort(gene_vars)[-50:]
            
            # Create correlation matrix
            if scipy.sparse.issparse(data.X):
                expr_matrix = data.X[:, top_genes_idx].todense()
            else:
                expr_matrix = data.X[:, top_genes_idx]
            
            corr_matrix = np.corrcoef(expr_matrix.T)
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                       vmin=-1, vmax=1, square=True)
            plt.title('Top 50 Variable Genes Correlation Matrix')
            plt.tight_layout()
            
            output_path = Path(self.output_dir)
            plt.savefig(output_path / "correlation_matrix.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.eda_results["visualizations"].append("correlation_matrix.png")
    
    def _perform_dim_reduction(self, data, split_desc):
        """Perform PCA and UMAP for visualization."""
        print("   🗺️  Performing dimensionality reduction...")
        
        try:
            import scanpy as sc
            
            # Make a copy for analysis
            adata_copy = data.copy()
            
            # Basic preprocessing
            sc.pp.normalize_total(adata_copy, target_sum=1e4)
            sc.pp.log1p(adata_copy)
            sc.pp.highly_variable_genes(adata_copy, min_mean=0.0125, max_mean=3, min_disp=0.5)
            adata_copy = adata_copy[:, adata_copy.var.highly_variable]
            
            # PCA
            sc.pp.scale(adata_copy, max_value=10)
            sc.tl.pca(adata_copy, svd_solver='arpack')
            
            # UMAP
            sc.pp.neighbors(adata_copy, n_neighbors=10, n_pcs=40)
            sc.tl.umap(adata_copy)
            
            # Plot PCA
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # PCA colored by age
            if 'age' in adata_copy.obs.columns:
                scatter = axes[0].scatter(
                    adata_copy.obsm['X_pca'][:, 0],
                    adata_copy.obsm['X_pca'][:, 1],
                    c=adata_copy.obs['age'],
                    cmap='viridis',
                    s=1,
                    alpha=0.6
                )
                axes[0].set_xlabel('PC1')
                axes[0].set_ylabel('PC2')
                axes[0].set_title('PCA - Colored by Age')
                plt.colorbar(scatter, ax=axes[0], label='Age (days)')
            
            # UMAP colored by genotype
            if 'genotype' in adata_copy.obs.columns:
                genotypes = adata_copy.obs['genotype'].unique()
                colors = plt.cm.Set1(np.linspace(0, 1, len(genotypes)))
                
                for i, genotype in enumerate(genotypes):
                    mask = adata_copy.obs['genotype'] == genotype
                    axes[1].scatter(
                        adata_copy.obsm['X_umap'][mask, 0],
                        adata_copy.obsm['X_umap'][mask, 1],
                        c=[colors[i]],
                        label=genotype,
                        s=1,
                        alpha=0.6
                    )
                
                axes[1].set_xlabel('UMAP1')
                axes[1].set_ylabel('UMAP2')
                axes[1].set_title('UMAP - Colored by Genotype')
                axes[1].legend()
            
            plt.tight_layout()
            output_path = Path(self.output_dir)
            fig.savefig(output_path / "dimensionality_reduction.png", dpi=150, bbox_inches='tight')
            plt.close()
            
            self.eda_results["visualizations"].append("dimensionality_reduction.png")
            
        except Exception as e:
            print(f"   ⚠️  Dimensionality reduction failed: {e}")
    
    def _analyze_gene_expression(self, data, split_desc):
        """Analyze gene expression patterns."""
        print("   🧬 Analyzing gene expression...")
        
        # Get expression matrix
        if scipy.sparse.issparse(data.X):
            expr_matrix = data.X.todense()
        else:
            expr_matrix = data.X
        
        # Calculate gene statistics
        gene_means = np.array(expr_matrix.mean(axis=0)).flatten()
        gene_vars = np.array(expr_matrix.var(axis=0)).flatten()
        
        # Find top expressed genes
        top_genes_idx = np.argsort(gene_means)[-20:]
        top_genes = data.var_names[top_genes_idx].tolist() if hasattr(data, 'var_names') else [f"Gene_{i}" for i in top_genes_idx]
        top_expressions = gene_means[top_genes_idx]
        
        # Create bar plot of top genes
        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = np.arange(len(top_genes))
        ax.barh(y_pos, top_expressions)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_genes)
        ax.set_xlabel('Mean Expression')
        ax.set_title('Top 20 Expressed Genes')
        plt.tight_layout()
        
        output_path = Path(self.output_dir)
        fig.savefig(output_path / "top_expressed_genes.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save gene statistics
        gene_stats_df = pd.DataFrame({
            'gene': data.var_names if hasattr(data, 'var_names') else [f"Gene_{i}" for i in range(len(gene_means))],
            'mean_expression': gene_means,
            'variance': gene_vars,
            'cv': gene_vars / (gene_means + 1e-10)  # Coefficient of variation
        })
        
        gene_stats_df = gene_stats_df.sort_values('mean_expression', ascending=False)
        gene_stats_df.head(100).to_csv(output_path / "top_genes.csv", index=False)
        
        self.eda_results["visualizations"].append("top_expressed_genes.png")
        self.eda_results["tables"]["top_genes"] = "top_genes.csv"
    
    def _save_eda_summary(self):
        """Save EDA summary to JSON."""
        summary_file = Path(self.output_dir) / "eda_summary.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "project": getattr(self.config, 'project', 'unknown'),
                "tissue": self.config.data.tissue,
                "batch_corrected": self.config.data.batch_correction.enabled,
            },
            "results": self.eda_results
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def generate_html_report(self, output_path=None):
        """
        Generate comprehensive HTML report with all EDA results.
        
        Args:
            output_path: Path to save HTML report
        """
        if output_path is None:
            output_path = Path(self.output_dir) / "eda_report.html"
        
        print(f"\n📝 Generating HTML report...")
        
        html_content = self._generate_html_content()
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"   ✅ Report saved to: {output_path}")
        
        return str(output_path)
    
    def _generate_html_content(self):
        """Generate HTML content for the report."""
        import base64
        
        # Embed images as base64
        embedded_images = {}
        for img_file in self.eda_results.get("visualizations", []):
            img_path = Path(self.output_dir) / img_file
            if img_path.exists():
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode()
                    embedded_images[img_file] = f"data:image/png;base64,{img_data}"
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TimeFlies EDA Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }}
        
        h1 {{
            margin: 0;
            font-size: 2.5rem;
        }}
        
        .subtitle {{
            opacity: 0.9;
            margin-top: 0.5rem;
        }}
        
        .section {{
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 0.5rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }}
        
        .stat-card {{
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 5px;
            border-left: 4px solid #667eea;
        }}
        
        .stat-value {{
            font-size: 1.8rem;
            font-weight: bold;
            color: #667eea;
        }}
        
        .stat-label {{
            font-size: 0.9rem;
            color: #666;
            text-transform: uppercase;
        }}
        
        .visualization {{
            margin: 2rem 0;
            text-align: center;
        }}
        
        .visualization img {{
            max-width: 100%;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        th, td {{
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        th {{
            background: #667eea;
            color: white;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .timestamp {{
            text-align: right;
            color: #666;
            font-size: 0.9rem;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>🔬 TimeFlies EDA Report</h1>
        <div class="subtitle">Comprehensive Exploratory Data Analysis</div>
    </header>
    
    <section class="section">
        <h2>📊 Dataset Overview</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('n_samples', 0):,}</div>
                <div class="stat-label">Total Samples</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('n_genes', 0):,}</div>
                <div class="stat-label">Total Genes</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('memory_mb', 0):.1f} MB</div>
                <div class="stat-label">Memory Usage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('sparsity', 0):.1%}</div>
                <div class="stat-label">Sparsity</div>
            </div>
        </div>
        
        {'<h3>Age Statistics</h3>' if 'age_mean' in self.eda_results['summary'] else ''}
        {'<div class="stats-grid">' if 'age_mean' in self.eda_results['summary'] else ''}
        {f'''
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('age_mean', 0):.1f}</div>
                <div class="stat-label">Mean Age (days)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('age_std', 0):.1f}</div>
                <div class="stat-label">Std Dev</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{self.eda_results['summary'].get('age_min', 0):.0f}-{self.eda_results['summary'].get('age_max', 0):.0f}</div>
                <div class="stat-label">Age Range</div>
            </div>
        ''' if 'age_mean' in self.eda_results['summary'] else ''}
        {'</div>' if 'age_mean' in self.eda_results['summary'] else ''}
    </section>
    
    <section class="section">
        <h2>📈 Visualizations</h2>
        
        {self._generate_visualization_section('age_distribution.png', 'Age Distribution', embedded_images)}
        {self._generate_visualization_section('correlation_matrix.png', 'Gene Correlation Matrix', embedded_images)}
        {self._generate_visualization_section('dimensionality_reduction.png', 'Dimensionality Reduction', embedded_images)}
        {self._generate_visualization_section('top_expressed_genes.png', 'Top Expressed Genes', embedded_images)}
    </section>
    
    <section class="section">
        <h2>📋 Data Tables</h2>
        <p>The following data files have been generated:</p>
        <ul>
            <li><strong>summary_stats.json</strong> - Complete statistical summary</li>
            <li><strong>top_genes.csv</strong> - Top 100 expressed genes with statistics</li>
            <li><strong>eda_summary.json</strong> - Full EDA results in JSON format</li>
        </ul>
    </section>
    
    <div class="timestamp">
        Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    </div>
</body>
</html>
"""
        return html
    
    def _generate_visualization_section(self, img_file, title, embedded_images):
        """Generate HTML for a visualization section."""
        if img_file in embedded_images:
            return f'''
        <div class="visualization">
            <h3>{title}</h3>
            <img src="{embedded_images[img_file]}" alt="{title}">
        </div>
            '''
        return ''
