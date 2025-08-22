"""
Utilities for generating smart names from split configurations.

This module handles the logic for creating compact, meaningful names
from column-based split configurations.
"""

from typing import List, Dict, Any, Union


class SplitNamingUtils:
    """
    Utilities for generating smart experiment names from split configurations.
    """
    
    # Common abbreviation mappings
    VALUE_ABBREVIATIONS = {
        "control": "ctrl",
        "ab42": "ab42",
        "htau": "htau", 
        "ab42htau": "ab42htau",
        "male": "m",
        "female": "f",
        "head": "h",
        "body": "b",
        "young": "y",
        "old": "o",
        "wildtype": "wt",
        "mutant": "mut",
    }
    
    # Smart grouping for common patterns
    GROUP_ABBREVIATIONS = {
        frozenset(["ab42", "htau"]): "alz",
        frozenset(["ab42", "htau", "ab42htau"]): "alz",
        frozenset(["male", "female"]): "all",
        frozenset(["head", "body"]): "all",
        frozenset([10, 20]): "young",
        frozenset([30, 40, 50]): "old",
    }

    @classmethod
    def abbreviate_values(cls, values: List[Union[str, int]]) -> str:
        """
        Create abbreviated name for a list of values.
        
        Args:
            values: List of values to abbreviate
            
        Returns:
            Abbreviated string representation
        """
        if not values:
            return "none"
            
        # Convert to lowercase strings for consistent matching
        normalized = [str(v).lower() for v in values]
        value_set = frozenset(normalized)
        
        # Check for group abbreviations first
        for group_set, abbrev in cls.GROUP_ABBREVIATIONS.items():
            group_normalized = frozenset(str(v).lower() for v in group_set)
            if value_set == group_normalized:
                return abbrev
                
        # Check for single value abbreviations
        if len(values) == 1:
            single_val = normalized[0]
            if single_val in cls.VALUE_ABBREVIATIONS:
                return cls.VALUE_ABBREVIATIONS[single_val]
                
        # Fallback: use first 2 values, abbreviated if possible
        abbreviated_parts = []
        for val in normalized[:2]:
            if val in cls.VALUE_ABBREVIATIONS:
                abbreviated_parts.append(cls.VALUE_ABBREVIATIONS[val])
            else:
                abbreviated_parts.append(val[:4])  # First 4 chars
                
        return "_".join(abbreviated_parts)

    @classmethod
    def generate_split_name(cls, split_config: Dict[str, Any]) -> str:
        """
        Generate a compact split name from configuration.
        
        Args:
            split_config: Split configuration dictionary
            
        Returns:
            Compact name like "ctrl-vs-alz" or "m-vs-f"
        """
        method = split_config.get("method", "random")
        
        if method == "random":
            # For random splits, just use "all" since entire dataset is used
            return "all"
            
        elif method == "column":
            train_values = split_config.get("train", [])
            test_values = split_config.get("test", [])
            
            train_abbrev = cls.abbreviate_values(train_values)
            test_abbrev = cls.abbreviate_values(test_values)
            
            return f"{train_abbrev}-vs-{test_abbrev}"
            
        else:
            return method  # Fallback

    @classmethod
    def generate_experiment_suffix(cls, config) -> str:
        """
        Generate experiment suffix from full config.
        
        Args:
            config: Configuration object
            
        Returns:
            Suffix string for experiment naming
        """
        # Get split name
        split_config = {
            "method": getattr(config.data.split, "method", "random"),
            "test_ratio": getattr(config.data.split, "test_ratio", 0.2),
            "train": getattr(config.data.split, "train", []),
            "test": getattr(config.data.split, "test", []),
        }
        
        split_name = cls.generate_split_name(split_config)
        
        # Add subset name (genes/cells/sex filters)
        subset_parts = []
        
        # Gene filtering
        if getattr(config.preprocessing.genes, "highly_variable_genes", False):
            subset_parts.append("hvg")
        elif getattr(config.preprocessing.genes, "remove_sex_genes", False):
            subset_parts.append("autogenes")
        elif getattr(config.preprocessing.genes, "only_keep_sex_genes", False):
            subset_parts.append("sexgenes")
            
        # Cell type filtering
        cell_type = getattr(config.data, "cell_type", "all")
        if cell_type != "all":
            subset_parts.append(cell_type[:4])  # First 4 chars
            
        # Sex filtering  
        sex_type = getattr(config.data, "sex_type", "all")
        if sex_type != "all":
            sex_abbrev = cls.VALUE_ABBREVIATIONS.get(sex_type, sex_type[:1])
            subset_parts.append(sex_abbrev)
            
        # Combine split name with subset
        if subset_parts:
            subset_suffix = "_".join(subset_parts)
            return f"{split_name}_{subset_suffix}"
        else:
            return split_name

    @classmethod 
    def extract_split_details_for_metadata(cls, config) -> Dict[str, Any]:
        """
        Extract detailed split information for metadata storage.
        
        Args:
            config: Configuration object
            
        Returns:
            Dictionary with split details for metadata
        """
        method = getattr(config.data.split, "method", "random")
        
        split_details = {
            "method": method,
            "split_name": cls.generate_split_name({
                "method": method,
                "test_ratio": getattr(config.data.split, "test_ratio", 0.2),
                "train": getattr(config.data.split, "train", []),
                "test": getattr(config.data.split, "test", []),
            })
        }
        
        if method == "column":
            split_details.update({
                "column": getattr(config.data.split, "column", None),
                "train_values": getattr(config.data.split, "train", []),
                "test_values": getattr(config.data.split, "test", []),
            })
        else:
            split_details.update({
                "test_ratio": getattr(config.data.split, "test_ratio", 0.2)
            })
            
        return split_details