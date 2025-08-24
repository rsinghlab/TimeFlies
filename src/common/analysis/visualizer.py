"""
Aging-Specific Visualization Module

Specialized plotting and visualization methods for aging research in Drosophila.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class AgingVisualizer:
    """
    Visualizer specialized for aging research plots.

    Provides methods for:
    - Age progression plots
    - Sex-specific aging visualizations
    - Aging marker expression plots
    - Trajectory visualizations
    """

    def __init__(self, config=None, style: str = "aging"):
        """
        Initialize aging visualizer.

        Args:
            config: Configuration object with plotting parameters
            style: Visual style ('aging', 'publication', 'presentation')
        """
        self.config = config
        self.style = style
        self._setup_plotting_style()

    def _setup_plotting_style(self):
        """Set up plotting style for aging visualizations."""
        if self.style == "aging":
            # Aging-appropriate color palette
            self.age_colors = {
                "young": "#2E8B57",  # Sea green
                "middle": "#FF8C00",  # Dark orange
                "old": "#DC143C",  # Crimson
                "very_old": "#4B0082",  # Indigo
            }
            self.sex_colors = {
                "male": "#4169E1",  # Royal blue
                "female": "#DC143C",  # Crimson
            }

        plt.style.use(
            "seaborn-v0_8-whitegrid"
            if hasattr(plt.style, "seaborn-v0_8-whitegrid")
            else "seaborn-whitegrid"
        )
        sns.set_palette("husl")

    def plot_age_progression(
        self,
        adata,
        genes: list[str],
        age_column: str = "age",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot gene expression changes across ages.

        Args:
            adata: AnnData object
            genes: List of genes to plot
            age_column: Column containing age information
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        n_genes = len(genes)
        fig, axes = plt.subplots(2, (n_genes + 1) // 2, figsize=(12, 8))
        if n_genes == 1:
            axes = [axes]
        axes = axes.flatten() if hasattr(axes, "flatten") else axes

        ages = sorted(adata.obs[age_column].unique())

        for i, gene in enumerate(genes):
            if gene not in adata.var.index:
                axes[i].text(
                    0.5,
                    0.5,
                    f"{gene}\nNot Found",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                continue

            gene_idx = list(adata.var.index).index(gene)
            expression = (
                adata.X[:, gene_idx].toarray().flatten()
                if hasattr(adata.X, "toarray")
                else adata.X[:, gene_idx]
            )

            # Create violin plot
            plot_data = []
            for age in ages:
                age_mask = adata.obs[age_column] == age
                age_expression = expression[age_mask]
                plot_data.extend([(age, expr) for expr in age_expression])

            plot_df = pd.DataFrame(plot_data, columns=["Age", "Expression"])

            sns.violinplot(data=plot_df, x="Age", y="Expression", ax=axes[i])
            axes[i].set_title(f"{gene} Expression by Age")
            axes[i].set_xlabel("Age (days)")
            axes[i].set_ylabel("Expression Level")

        # Remove unused subplots
        for i in range(n_genes, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_sex_aging_comparison(
        self,
        adata,
        genes: list[str],
        age_column: str = "age",
        sex_column: str = "sex",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot sex-specific aging patterns.

        Args:
            adata: AnnData object
            genes: List of genes to plot
            age_column: Column containing age information
            sex_column: Column containing sex information
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        n_genes = len(genes)
        fig, axes = plt.subplots(1, n_genes, figsize=(4 * n_genes, 6))
        if n_genes == 1:
            axes = [axes]

        ages = sorted(adata.obs[age_column].unique())

        for i, gene in enumerate(genes):
            if gene not in adata.var.index:
                axes[i].text(
                    0.5,
                    0.5,
                    f"{gene}\nNot Found",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                continue

            gene_idx = list(adata.var.index).index(gene)

            # Plot for each sex
            for sex in ["male", "female"]:
                sex_data = adata[adata.obs[sex_column] == sex]
                if len(sex_data) == 0:
                    continue

                sex_ages = []
                sex_expression = []
                sex_std = []

                for age in ages:
                    age_sex_data = sex_data[sex_data.obs[age_column] == age]
                    if len(age_sex_data) > 0:
                        expr = (
                            age_sex_data.X[:, gene_idx].toarray().flatten()
                            if hasattr(age_sex_data.X, "toarray")
                            else age_sex_data.X[:, gene_idx]
                        )
                        sex_ages.append(age)
                        sex_expression.append(np.mean(expr))
                        sex_std.append(np.std(expr))

                # Plot with error bars
                axes[i].errorbar(
                    sex_ages,
                    sex_expression,
                    yerr=sex_std,
                    label=sex.capitalize(),
                    color=self.sex_colors[sex],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                )

            axes[i].set_title(f"{gene}: Sex-Specific Aging")
            axes[i].set_xlabel("Age (days)")
            axes[i].set_ylabel("Mean Expression Level")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_aging_markers_heatmap(
        self,
        adata,
        aging_markers: list[str],
        age_column: str = "age",
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Create heatmap of aging marker expression across ages.

        Args:
            adata: AnnData object
            aging_markers: List of aging marker genes
            age_column: Column containing age information
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        ages = sorted(adata.obs[age_column].unique())
        available_markers = [gene for gene in aging_markers if gene in adata.var.index]

        if not available_markers:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No aging markers found in data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # Create expression matrix: markers x ages
        expr_matrix = np.zeros((len(available_markers), len(ages)))

        for i, marker in enumerate(available_markers):
            marker_idx = list(adata.var.index).index(marker)

            for j, age in enumerate(ages):
                age_data = adata[adata.obs[age_column] == age]
                if len(age_data) > 0:
                    expr = (
                        age_data.X[:, marker_idx].toarray().flatten()
                        if hasattr(age_data.X, "toarray")
                        else age_data.X[:, marker_idx]
                    )
                    expr_matrix[i, j] = np.mean(expr)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(
            expr_matrix,
            xticklabels=[f"Age {age}" for age in ages],
            yticklabels=available_markers,
            annot=True,
            fmt=".2f",
            cmap="RdYlBu_r",
            ax=ax,
        )

        ax.set_title("Aging Markers Expression Across Ages", fontsize=16, pad=20)
        ax.set_xlabel("Age Groups", fontsize=12)
        ax.set_ylabel("Aging Markers", fontsize=12)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def plot_trajectory_genes(
        self,
        trajectory_results: dict,
        adata,
        age_column: str = "age",
        top_n: int = 6,
        save_path: Path | None = None,
    ) -> plt.Figure:
        """
        Plot top trajectory genes over age.

        Args:
            trajectory_results: Results from trajectory analysis
            adata: AnnData object
            age_column: Column containing age information
            top_n: Number of top genes to plot
            save_path: Optional path to save the plot

        Returns:
            matplotlib Figure object
        """
        significant_trajectories = trajectory_results["significant_trajectories"]

        if not significant_trajectories:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(
                0.5,
                0.5,
                "No significant trajectories found",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=16,
            )
            return fig

        # Sort by significance (p-value for linear, R² for polynomial)
        method = trajectory_results["method"]
        if method == "linear":
            sorted_genes = sorted(
                significant_trajectories.items(), key=lambda x: x[1]["p_value"]
            )[:top_n]
        else:
            sorted_genes = sorted(
                significant_trajectories.items(),
                key=lambda x: x[1]["r_squared"],
                reverse=True,
            )[:top_n]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        ages = sorted(adata.obs[age_column].unique())

        for i, (gene, trajectory_data) in enumerate(sorted_genes):
            if gene not in adata.var.index:
                continue

            gene_idx = list(adata.var.index).index(gene)

            # Get expression data
            plot_ages = []
            plot_expression = []

            for age in ages:
                age_data = adata[adata.obs[age_column] == age]
                if len(age_data) > 0:
                    expr = (
                        age_data.X[:, gene_idx].toarray().flatten()
                        if hasattr(age_data.X, "toarray")
                        else age_data.X[:, gene_idx]
                    )
                    plot_ages.extend([age] * len(expr))
                    plot_expression.extend(expr)

            # Scatter plot
            axes[i].scatter(plot_ages, plot_expression, alpha=0.6, s=20)

            # Fit line
            if method == "linear":
                z = np.polyfit(plot_ages, plot_expression, 1)
                p = np.poly1d(z)
                axes[i].plot(ages, p(ages), "r--", linewidth=2)

                title = f"{gene}\n(r={trajectory_data['correlation']:.3f}, p={trajectory_data['p_value']:.2e})"
            else:
                coeffs = trajectory_data["coefficients"]
                p = np.poly1d(coeffs)
                age_range = np.linspace(min(ages), max(ages), 100)
                axes[i].plot(age_range, p(age_range), "r--", linewidth=2)

                title = f"{gene}\n(R²={trajectory_data['r_squared']:.3f}, {trajectory_data['trajectory_type']})"

            axes[i].set_title(title, fontsize=10)
            axes[i].set_xlabel("Age (days)")
            axes[i].set_ylabel("Expression Level")
            axes[i].grid(True, alpha=0.3)

        # Remove unused subplots
        for i in range(len(sorted_genes), len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig
