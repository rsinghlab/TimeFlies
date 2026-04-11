"""Tests for analysis modules (EDA and visualization functionality)."""

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from timeflies.analysis.eda import EDAHandler


class TestEDAHandler:
    """Test EDAHandler functionality."""

    def setup_method(self):
        """Set up test environment for each test."""
        self.adata = self._create_sample_adata()

    def _create_sample_adata(self) -> AnnData:
        """Create a small sample AnnData object for testing."""
        n_obs, n_vars = 100, 200

        X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars)).astype(np.float32)

        obs = pd.DataFrame(
            {
                "age": np.random.choice([1, 5, 10, 20], n_obs),
                "sex": np.random.choice(["male", "female"], n_obs),
                "tissue": np.random.choice(["head", "body"], n_obs),
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )

        var = pd.DataFrame(
            {"gene_type": np.random.choice(["protein_coding", "lncRNA"], n_vars)},
            index=[f"gene_{i}" for i in range(n_vars)],
        )

        return AnnData(X=X, obs=obs, var=var)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
