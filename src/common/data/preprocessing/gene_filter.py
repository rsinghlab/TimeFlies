"""Gene filtering utilities for preprocessing genomic data."""

from typing import Any, Optional

import numpy as np
from anndata import AnnData


class GeneFilter:
    """
    A class to handle gene filtering based on configuration settings.

    This class takes in multiple datasets and gene lists, and applies filters based on
    the configuration provided.
    """

    def __init__(
        self,
        config: Any,
        adata: AnnData,
        adata_eval: AnnData,
        adata_original: AnnData,
        autosomal_genes: list[str],
        sex_genes: list[str],
    ):
        """
        Initializes the GeneFilter with the given configuration and datasets.

        Parameters:
        - config: The configuration object containing all necessary parameters.
        - adata: The main AnnData object containing the dataset.
        - adata_eval: The AnnData object for evaluation data.
        - adata_original: The original unfiltered AnnData object.
        - autosomal_genes: A list of autosomal genes.
        - sex_genes: A list of sex-linked genes.
        """
        self.config = config
        self.adata = adata
        self.adata_eval = adata_eval
        self.adata_original = adata_original
        self.autosomal_genes = autosomal_genes
        self.sex_genes = sex_genes

    def balance_genes(
        self, gene_type_1: list[str], gene_type_2: list[str]
    ) -> list[str]:
        """
        Balance the number of genes in gene_type_1 to match the number of genes in gene_type_2.

        Parameters:
        gene_type_1: List of genes of the first type (e.g., autosomal genes).
        gene_type_2: List of genes of the second type (e.g., X genes).

        Returns:
        balanced_genes: Subset of gene_type_1 with the same size as the number of genes in gene_type_2.
        """
        config = self.config
        random_state = getattr(config.general, "random_state", 42)
        np.random.seed(random_state)
        num_genes_type_2 = len(gene_type_2)
        if num_genes_type_2 > len(gene_type_1):
            raise ValueError(
                "Number of genes in gene_type_2 is greater than the number of genes in gene_type_1 available."
            )

        # Select a random subset of gene_type_1 to match the size of gene_type_2
        balanced_genes = np.random.choice(
            gene_type_1, num_genes_type_2, replace=False
        ).tolist()
        return balanced_genes

    def create_and_apply_mask(
        self,
        data: AnnData,
        sex_genes: list[str],
        autosomal_genes: list[str],
        original_autosomal_genes: list[str],
        balanced_non_lnc_genes: list[str] | None = None,
    ) -> AnnData:
        """
        Create and apply masks to filter genes in the dataset based on configuration.

        Parameters:
        data: Annotated data matrix with genes as variables.
        sex_genes: List of sex-linked genes.
        autosomal_genes: List of autosomal genes after balancing (if applicable).
        original_autosomal_genes: Original list of autosomal genes before any balancing.
        balanced_non_lnc_genes: List of non-lncRNA genes used in balancing. Defaults to None.

        Returns:
        data: Filtered data matrix based on the applied masks.
        """
        config = self.config

        # Create masks for sex genes and autosomal genes
        sex_mask = data.var.index.isin(sex_genes)
        autosomal_mask = data.var.index.isin(autosomal_genes)
        original_autosomal_mask = data.var.index.isin(original_autosomal_genes)

        # Create lncRNA mask
        lnc_mask = data.var_names.str.startswith("lnc")
        no_lnc_mask = ~lnc_mask

        # Create non-lncRNA mask (if balanced_non_lnc_genes is provided)
        if balanced_non_lnc_genes is not None:
            balanced_non_lnc_mask = data.var.index.isin(balanced_non_lnc_genes)

        # Remove unaccounted genes based on the original set of autosomal and sex genes
        if getattr(config.preprocessing.genes, "remove_unaccounted_genes", False):
            accounted_mask = original_autosomal_mask | sex_mask
            data = data[:, accounted_mask]
            # Recompute the masks since data has changed
            sex_mask = data.var.index.isin(sex_genes)
            autosomal_mask = data.var.index.isin(autosomal_genes)
            original_autosomal_mask = data.var.index.isin(original_autosomal_genes)
            lnc_mask = data.var_names.str.startswith("lnc")
            no_lnc_mask = ~lnc_mask
            if balanced_non_lnc_genes is not None:
                balanced_non_lnc_mask = data.var.index.isin(balanced_non_lnc_genes)
        else:
            # Include genes not found in the provided gene lists
            unaccounted_mask = ~(original_autosomal_mask | sex_mask)

        # Initialize final_mask as all True
        final_mask = np.ones(data.shape[1], dtype=bool)

        # Apply various filters based on configuration settings
        remove_sex = getattr(config.preprocessing.genes, "remove_sex_genes", False)
        balance_genes = getattr(config.preprocessing.balancing, "balance_genes", False)
        balance_lnc_genes = getattr(
            config.preprocessing.balancing, "balance_lnc_genes", False
        )
        only_keep_lnc = getattr(
            config.preprocessing.genes, "only_keep_lnc_genes", False
        )
        remove_autosomal = getattr(
            config.preprocessing.genes, "remove_autosomal_genes", False
        )
        remove_lnc = getattr(config.preprocessing.genes, "remove_lnc_genes", False)
        remove_unaccounted = getattr(
            config.preprocessing.genes, "remove_unaccounted_genes", False
        )
        balance_autosomal = remove_sex and balance_genes

        if balance_autosomal:
            final_mask &= autosomal_mask

        elif balance_lnc_genes:
            final_mask &= balanced_non_lnc_mask

        else:
            if only_keep_lnc:
                final_mask &= lnc_mask

            if remove_autosomal:
                final_mask &= ~autosomal_mask

            if remove_sex:
                final_mask &= ~sex_mask

            if remove_lnc:
                final_mask &= no_lnc_mask

        # If not removing unaccounted genes, ensure they are included in the final mask
        if (
            not remove_unaccounted
            and balanced_non_lnc_genes is None
            and not only_keep_lnc
            and not balance_autosomal
        ):
            final_mask |= unaccounted_mask

        # Apply the final combined mask
        data = data[:, final_mask]

        return data

    def filter_genes_based_on_config(
        self,
        adata: AnnData,
        adata_eval: AnnData,
        adata_original: AnnData,
        sex_genes: list[str],
        autosomal_genes: list[str],
    ) -> tuple[AnnData, AnnData, AnnData]:
        """
        Filter genes in multiple datasets based on provided configurations.

        Parameters:
        adata: Annotated data matrix for primary analysis.
        adata_eval: Annotated data matrix for evaluation purposes.
        adata_original: Original annotated data matrix for reference.
        sex_genes: List of sex-linked genes.
        autosomal_genes: List of autosomal genes.

        Returns:
        Tuple[AnnData, AnnData, AnnData]: Filtered versions of adata, adata_eval, and adata_original.
        """
        config = self.config

        # Derive lncRNA genes and non-lncRNA genes
        lnc_genes = adata.var_names[adata.var_names.str.startswith("lnc")].tolist()
        non_lnc_genes = [gene for gene in adata.var.index if not gene.startswith("lnc")]

        # Store the original autosomal genes before any balancing
        original_autosomal_genes = autosomal_genes.copy()

        # Balance the number of autosomal genes with the number of X genes if required
        balance_genes = getattr(config.preprocessing.balancing, "balance_genes", False)
        if balance_genes:
            autosomal_genes = self.balance_genes(autosomal_genes, sex_genes)

        # Balance the number of non-lnc genes with the number of lnc genes if required
        balance_lnc_genes = getattr(
            config.preprocessing.balancing, "balance_lnc_genes", False
        )
        balanced_non_lnc_genes = None
        if balance_lnc_genes:
            balanced_non_lnc_genes = self.balance_genes(non_lnc_genes, lnc_genes)

        # Apply the balanced genes masks to the datasets
        adata = self.create_and_apply_mask(
            data=adata,
            sex_genes=sex_genes,
            autosomal_genes=autosomal_genes,
            original_autosomal_genes=original_autosomal_genes,
            balanced_non_lnc_genes=balanced_non_lnc_genes
            if balance_lnc_genes
            else None,
        )
        adata_eval = self.create_and_apply_mask(
            data=adata_eval,
            sex_genes=sex_genes,
            autosomal_genes=autosomal_genes,
            original_autosomal_genes=original_autosomal_genes,
            balanced_non_lnc_genes=balanced_non_lnc_genes
            if balance_lnc_genes
            else None,
        )
        adata_original = self.create_and_apply_mask(
            data=adata_original,
            sex_genes=sex_genes,
            autosomal_genes=autosomal_genes,
            original_autosomal_genes=original_autosomal_genes,
            balanced_non_lnc_genes=balanced_non_lnc_genes
            if balance_lnc_genes
            else None,
        )

        return adata, adata_eval, adata_original

    def apply_filter(self) -> tuple[AnnData, AnnData, AnnData]:
        """
        Applies gene filters based on the configuration settings.

        This method filters the genes in the datasets according to the configuration
        settings, potentially removing autosomal genes, sex-linked genes, or other
        specified gene groups.

        Returns:
        - tuple: A tuple containing the filtered AnnData objects for the main, evaluation,
                 and original datasets.
        """
        # Apply the gene filtering based on the config
        (
            self.adata,
            self.adata_eval,
            self.adata_original,
        ) = self.filter_genes_based_on_config(
            self.adata,
            self.adata_eval,
            self.adata_original,
            self.sex_genes,
            self.autosomal_genes,
        )
        return self.adata, self.adata_eval, self.adata_original
