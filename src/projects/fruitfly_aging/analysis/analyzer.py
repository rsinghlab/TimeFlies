"""
Aging-Specific Analysis Module

Specialized analysis methods for studying aging in Drosophila melanogaster.
Builds on the core TimeFlies framework with aging-specific insights.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from pathlib import Path


class AgingAnalyzer:
    """
    Analyzer specialized for aging research in fruit flies.

    Provides methods for:
    - Age-related differential expression analysis
    - Aging trajectory modeling
    - Sex-specific aging patterns
    - Cross-tissue aging comparisons
    """

    def __init__(self, config=None):
        """
        Initialize aging analyzer.

        Args:
            config: Configuration object with aging-specific parameters
        """
        self.config = config
        self.age_groups = self._define_age_groups()
        self.aging_markers = self._load_aging_markers()

    def _define_age_groups(self) -> Dict[str, List[int]]:
        """Define age groups based on actual data: 5, 30, 50, 70 days."""
        return {"young": [5], "middle": [30], "old": [50], "very_old": [70]}

    def _load_aging_markers(self) -> List[str]:
        """Load known aging marker genes for Drosophila."""
        # These could be loaded from a file in the future
        return [
            "InR",  # Insulin receptor
            "foxo",  # Forkhead box O
            "sod1",  # Superoxide dismutase 1
            "cat",  # Catalase
            "hsp70",  # Heat shock protein 70
            "tor",  # Target of rapamycin
            "sir2",  # Sirtuin 2
            "p53",  # Tumor suppressor p53
        ]

    def analyze_age_progression(self, adata, age_column: str = "age") -> Dict:
        """
        Analyze gene expression changes across age groups.

        Args:
            adata: AnnData object with age information
            age_column: Column name containing age data

        Returns:
            Dictionary with age progression analysis results
        """
        results = {}
        ages = sorted(adata.obs[age_column].unique())

        # Correlation with age
        age_correlations = self._compute_age_correlations(adata, age_column)
        results["age_correlations"] = age_correlations

        # Differential expression between age groups
        de_results = {}
        for i, age1 in enumerate(ages[:-1]):
            age2 = ages[i + 1]
            de_genes = self._differential_expression_age_groups(
                adata, age1, age2, age_column
            )
            de_results[f"{age1}vs{age2}"] = de_genes

        results["differential_expression"] = de_results

        # Aging marker expression
        marker_expression = self._analyze_aging_markers(adata, age_column)
        results["aging_markers"] = marker_expression

        return results

    def analyze_sex_specific_aging(
        self, adata, age_column: str = "age", sex_column: str = "sex"
    ) -> Dict:
        """
        Analyze sex-specific patterns in aging.

        Args:
            adata: AnnData object with age and sex information
            age_column: Column name containing age data
            sex_column: Column name containing sex data

        Returns:
            Dictionary with sex-specific aging analysis
        """
        results = {}

        # Separate male and female data
        male_data = adata[adata.obs[sex_column] == "male"]
        female_data = adata[adata.obs[sex_column] == "female"]

        # Age progression analysis for each sex
        results["male_aging"] = self.analyze_age_progression(male_data, age_column)
        results["female_aging"] = self.analyze_age_progression(female_data, age_column)

        # Sex differences at each age
        sex_differences = {}
        for age in adata.obs[age_column].unique():
            age_data = adata[adata.obs[age_column] == age]
            if len(age_data.obs[sex_column].unique()) == 2:  # Both sexes present
                sex_de = self._differential_expression_sex(age_data, sex_column)
                sex_differences[f"age_{age}"] = sex_de

        results["sex_differences_by_age"] = sex_differences

        return results

    def identify_aging_trajectories(
        self, adata, age_column: str = "age", method: str = "polynomial"
    ) -> Dict:
        """
        Identify genes with significant aging trajectories.

        Args:
            adata: AnnData object
            age_column: Column name containing age data
            method: Method for trajectory fitting ('linear', 'polynomial')

        Returns:
            Dictionary with trajectory analysis results
        """
        import scipy.stats as stats

        ages = adata.obs[age_column].values
        trajectories = {}

        for gene_idx, gene in enumerate(adata.var.index):
            expression = (
                adata.X[:, gene_idx].toarray().flatten()
                if hasattr(adata.X, "toarray")
                else adata.X[:, gene_idx]
            )

            if method == "linear":
                # Linear correlation with age
                correlation, p_value = stats.pearsonr(ages, expression)
                trajectories[gene] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "trajectory_type": "increasing"
                    if correlation > 0
                    else "decreasing",
                }

            elif method == "polynomial":
                # Polynomial fit to identify non-linear patterns
                coeffs = np.polyfit(ages, expression, 2)
                fit = np.polyval(coeffs, ages)
                r_squared = 1 - (
                    np.sum((expression - fit) ** 2)
                    / np.sum((expression - np.mean(expression)) ** 2)
                )

                trajectories[gene] = {
                    "coefficients": coeffs,
                    "r_squared": r_squared,
                    "trajectory_type": self._classify_polynomial_trajectory(coeffs),
                }

        # Sort by significance/fit quality
        if method == "linear":
            significant_genes = {
                k: v for k, v in trajectories.items() if v["p_value"] < 0.05
            }
        else:
            significant_genes = {
                k: v for k, v in trajectories.items() if v["r_squared"] > 0.1
            }

        return {
            "all_trajectories": trajectories,
            "significant_trajectories": significant_genes,
            "method": method,
        }

    def _compute_age_correlations(self, adata, age_column: str) -> Dict:
        """Compute correlations between gene expression and age."""
        import scipy.stats as stats

        ages = adata.obs[age_column].values
        correlations = {}

        for gene_idx, gene in enumerate(adata.var.index):
            expression = (
                adata.X[:, gene_idx].toarray().flatten()
                if hasattr(adata.X, "toarray")
                else adata.X[:, gene_idx]
            )
            correlation, p_value = stats.pearsonr(ages, expression)
            correlations[gene] = {"correlation": correlation, "p_value": p_value}

        return correlations

    def _differential_expression_age_groups(
        self, adata, age1: int, age2: int, age_column: str
    ) -> Dict:
        """Perform differential expression between two age groups."""
        # Placeholder for differential expression analysis
        # In practice, this would use scanpy or other DE methods
        group1 = adata[adata.obs[age_column] == age1]
        group2 = adata[adata.obs[age_column] == age2]

        return {
            "comparison": f"{age1}_vs_{age2}",
            "group1_size": len(group1),
            "group2_size": len(group2),
            "note": "Differential expression analysis would be implemented here",
        }

    def _differential_expression_sex(self, adata, sex_column: str) -> Dict:
        """Perform differential expression between sexes."""
        male_data = adata[adata.obs[sex_column] == "male"]
        female_data = adata[adata.obs[sex_column] == "female"]

        return {
            "comparison": "male_vs_female",
            "male_cells": len(male_data),
            "female_cells": len(female_data),
            "note": "Sex-specific differential expression analysis would be implemented here",
        }

    def _analyze_aging_markers(self, adata, age_column: str) -> Dict:
        """Analyze expression of known aging marker genes."""
        marker_results = {}

        for marker in self.aging_markers:
            if marker in adata.var.index:
                marker_idx = list(adata.var.index).index(marker)
                expression = (
                    adata.X[:, marker_idx].toarray().flatten()
                    if hasattr(adata.X, "toarray")
                    else adata.X[:, marker_idx]
                )

                # Expression by age group
                age_expression = {}
                for age in adata.obs[age_column].unique():
                    age_cells = adata.obs[age_column] == age
                    age_expression[str(age)] = {
                        "mean": float(np.mean(expression[age_cells])),
                        "std": float(np.std(expression[age_cells])),
                        "n_cells": int(np.sum(age_cells)),
                    }

                marker_results[marker] = age_expression

        return marker_results

    def _classify_polynomial_trajectory(self, coeffs: np.ndarray) -> str:
        """Classify polynomial trajectory based on coefficients."""
        a, b, c = coeffs

        if abs(a) < 1e-6:  # Essentially linear
            return "linear_increasing" if b > 0 else "linear_decreasing"
        elif a > 0:
            return "u_shaped"  # Decreases then increases
        else:
            return "inverted_u"  # Increases then decreases
