"""
Notebook-friendly interface for TimeFlies.

This module provides a clean, object-oriented interface designed for Jupyter notebooks
and interactive Python use.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict
import pandas as pd


@dataclass
class Config:
    """
    TimeFlies configuration for notebook use.
    
    Auto-detects existing YAML configs or creates new ones in notebook.
    
    Examples:
        # Use existing YAML config
        >>> config = Config.from_yaml("configs/default.yaml")
        
        # Create new config in notebook  
        >>> config = Config(
        ...     project="alzheimers",
        ...     tissue="head",
        ...     model="cnn",
        ...     target="age"
        ... )
        
        # Auto-detect notebook config
        >>> config = Config.auto()  # Looks for notebook_config.json or uses defaults
    """
    # Core settings
    project: str = "fruitfly_alzheimers"  # fruitfly_aging, fruitfly_alzheimers
    tissue: str = "head"                  # head, body, all
    model: str = "cnn"                    # cnn, mlp, xgboost, random_forest, logistic
    target: str = "age"                   # target variable
    
    # Data settings  
    batch_corrected: bool = False         # Use batch-corrected data
    samples: Optional[int] = 10000        # Number of samples (None for all)
    variables: Optional[int] = None       # Number of genes (None for all)
    
    # Training settings
    epochs: int = 100
    batch_size: int = 32
    validation_split: float = 0.2
    
    # Split configuration
    split_method: str = "column"          # column, random
    split_column: str = "genotype"        # column to split on
    train_values: List[str] = None        # values for training
    test_values: List[str] = None         # values for testing
    
    def __post_init__(self):
        """Set defaults based on project."""
        if self.train_values is None:
            if self.project == "fruitfly_alzheimers":
                self.train_values = ["control"]
                self.test_values = ["ab42", "htau"] if self.test_values is None else self.test_values
            else:
                self.train_values = ["young"]  
                self.test_values = ["old"] if self.test_values is None else self.test_values
    
    @classmethod
    def auto(cls) -> 'Config':
        """
        Auto-detect configuration from notebook directory.
        
        Looks for:
        1. notebook_config.json (created by Config.save())
        2. configs/default.yaml (TimeFlies YAML config)
        3. Falls back to defaults
        
        Returns:
            Config: Auto-detected or default configuration
        """
        # Try notebook JSON config first
        if Path("notebook_config.json").exists():
            return cls.load("notebook_config.json")
            
        # Try TimeFlies YAML config
        for yaml_path in ["configs/default.yaml", "../configs/default.yaml", "../../configs/default.yaml"]:
            if Path(yaml_path).exists():
                return cls.from_yaml(yaml_path)
        
        # Default config
        print("🔧 Using default configuration. Save with config.save() to persist.")
        return cls()
    
    @classmethod  
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from TimeFlies YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            Config: Configuration loaded from YAML
        """
        try:
            import yaml
            
            with open(yaml_path, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # Extract relevant settings
            config_data = {
                "project": yaml_data.get("project", "fruitfly_alzheimers"),
                "tissue": yaml_data.get("data", {}).get("tissue", "head"),
                "model": yaml_data.get("data", {}).get("model", "cnn").lower(),
                "target": yaml_data.get("data", {}).get("target_variable", "age"),
                "batch_corrected": yaml_data.get("data", {}).get("batch_correction", {}).get("enabled", False),
                "samples": yaml_data.get("data", {}).get("sampling", {}).get("samples"),
                "variables": yaml_data.get("data", {}).get("sampling", {}).get("variables"),
                "epochs": yaml_data.get("model", {}).get("training", {}).get("epochs", 100),
                "batch_size": yaml_data.get("model", {}).get("training", {}).get("batch_size", 32),
                "validation_split": yaml_data.get("model", {}).get("training", {}).get("validation_split", 0.2),
                "split_method": yaml_data.get("data", {}).get("split", {}).get("method", "column"),
                "split_column": yaml_data.get("data", {}).get("split", {}).get("column", "genotype"),
                "train_values": yaml_data.get("data", {}).get("split", {}).get("train", []),
                "test_values": yaml_data.get("data", {}).get("split", {}).get("test", []),
            }
            
            print(f"✅ Loaded configuration from {yaml_path}")
            return cls(**{k: v for k, v in config_data.items() if v is not None})
            
        except ImportError:
            print("⚠️  PyYAML not available. Install with: pip install pyyaml")
            return cls()
        except Exception as e:
            print(f"⚠️  Failed to load YAML config: {e}")
            return cls()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CLI use."""
        return asdict(self)
    
    def save(self, path: str = "notebook_config.json"):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
    @classmethod
    def load(cls, path: str = "notebook_config.json") -> 'Config':
        """Load config from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)


class TimeFliesResult:
    """Results container for TimeFlies operations."""
    
    def __init__(self, 
                 config: Config,
                 success: bool = True,
                 experiment_dir: Optional[str] = None,
                 metrics: Optional[Dict] = None,
                 predictions: Optional[pd.DataFrame] = None,
                 error: Optional[str] = None):
        self.config = config
        self.success = success
        self.experiment_dir = experiment_dir
        self.metrics = metrics or {}
        self.predictions = predictions
        self.error = error
        self._html_report = None
    
    @property
    def mae(self) -> float:
        """Mean absolute error."""
        return self.metrics.get("mae", 0.0)
    
    @property 
    def r2(self) -> float:
        """R-squared score."""
        return self.metrics.get("r2_score", 0.0)
        
    @property
    def correlation(self) -> float:
        """Pearson correlation."""
        return self.metrics.get("pearson_correlation", 0.0)
        
    def summary(self) -> str:
        """Get summary string."""
        if not self.success:
            return f"❌ Failed: {self.error}"
            
        return f"""
✅ TimeFlies Results Summary
============================
Project: {self.config.project}
Model: {self.config.model.upper()}  
Target: {self.config.target}

Performance:
  MAE: {self.mae:.3f}
  R²: {self.r2:.3f}  
  Correlation: {self.correlation:.3f}

Experiment: {Path(self.experiment_dir).name if self.experiment_dir else 'N/A'}
"""

    def __repr__(self) -> str:
        return self.summary()
    
    def show_predictions(self, n: int = 10) -> pd.DataFrame:
        """Show prediction results."""
        if self.predictions is None:
            print("No predictions available")
            return None
        return self.predictions.head(n)
    
    def plot_results(self):
        """Plot prediction results."""
        if self.predictions is None:
            print("No predictions available")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot
            axes[0].scatter(self.predictions['actual_age'], 
                          self.predictions['predicted_age'], 
                          alpha=0.6, s=1)
            axes[0].plot([self.predictions['actual_age'].min(), 
                         self.predictions['actual_age'].max()],
                        [self.predictions['actual_age'].min(), 
                         self.predictions['actual_age'].max()], 
                        'r--', alpha=0.8)
            axes[0].set_xlabel('Actual Age')
            axes[0].set_ylabel('Predicted Age')
            axes[0].set_title('Predicted vs Actual')
            
            # Error distribution  
            axes[1].hist(self.predictions['prediction_error'], bins=30, alpha=0.7)
            axes[1].set_xlabel('Prediction Error')
            axes[1].set_ylabel('Count')
            axes[1].set_title('Error Distribution')
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting")


def _run_timeflies_command(command: List[str], config: Optional[Config] = None) -> TimeFliesResult:
    """Run a TimeFlies CLI command and return results."""
    try:
        # Get project root
        project_root = Path(__file__).parent.parent.parent
        
        # Build full command
        full_command = [sys.executable, str(project_root / "run_timeflies.py")]
        
        # Add config overrides
        if config:
            if config.project == "fruitfly_alzheimers":
                full_command.append("--alzheimers")
            else:
                full_command.append("--aging")
                
            if config.batch_corrected:
                full_command.append("--batch-corrected")
                
            if config.tissue != "head":
                full_command.extend(["--tissue", config.tissue])
                
            if config.model != "cnn":
                full_command.extend(["--model", config.model])
                
            if config.target != "age":
                full_command.extend(["--target", config.target])
        
        # Add command
        full_command.extend(command)
        
        # Set environment
        env = {"PYTHONPATH": str(project_root / "src")}
        
        # Run command
        result = subprocess.run(
            full_command,
            cwd=project_root,
            env={**os.environ, **env},
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            return TimeFliesResult(
                config=config or Config(),
                success=False,
                error=result.stderr or result.stdout
            )
        
        # Parse results from output
        experiment_dir = None
        for line in result.stdout.split('\n'):
            if 'Results saved to:' in line:
                experiment_dir = line.split('Results saved to:')[-1].strip()
                break
            elif 'Experiment:' in line:
                exp_name = line.split('Experiment:')[-1].strip()
                # Construct full path
                if config:
                    correction = "batch_corrected" if config.batch_corrected else "uncorrected"
                    experiment_dir = f"outputs/{config.project}/experiments/{correction}/{exp_name}"
        
        # Load metrics if available
        metrics = {}
        predictions = None
        
        if experiment_dir and Path(experiment_dir).exists():
            metrics_file = Path(experiment_dir) / "evaluation" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
            
            predictions_file = Path(experiment_dir) / "evaluation" / "predictions.csv"
            if predictions_file.exists():
                predictions = pd.read_csv(predictions_file)
        
        return TimeFliesResult(
            config=config or Config(),
            success=True,
            experiment_dir=experiment_dir,
            metrics=metrics,
            predictions=predictions
        )
        
    except Exception as e:
        return TimeFliesResult(
            config=config or Config(),
            success=False,
            error=str(e)
        )


def train(config: Config, with_eda: bool = False, with_analysis: bool = False) -> TimeFliesResult:
    """
    Train a model with the given configuration.
    
    Args:
        config: TimeFlies configuration
        with_eda: Run EDA before training
        with_analysis: Run analysis after training
        
    Returns:
        TimeFliesResult: Training results
        
    Example:
        >>> config = Config(project="alzheimers", model="cnn")
        >>> results = train(config, with_eda=True)
        >>> print(results.mae)  # Mean absolute error
        >>> results.plot_results()  # Show plots
    """
    command = ["train"]
    
    if with_eda:
        command.append("--with-eda")
    if with_analysis:
        command.append("--with-analysis")
    
    return _run_timeflies_command(command, config)


def evaluate(config: Config, with_eda: bool = False, with_analysis: bool = False) -> TimeFliesResult:
    """
    Evaluate an existing model with the given configuration.
    
    Args:
        config: TimeFlies configuration  
        with_eda: Run EDA before evaluation
        with_analysis: Run analysis after evaluation
        
    Returns:
        TimeFliesResult: Evaluation results
    """
    command = ["evaluate"]
    
    if with_eda:
        command.append("--with-eda")
    if with_analysis:
        command.append("--with-analysis")
    
    return _run_timeflies_command(command, config)


def eda(config: Config, 
        split: str = "all",
        save_report: bool = True) -> TimeFliesResult:
    """
    Run exploratory data analysis.
    
    Args:
        config: TimeFlies configuration
        split: Which data split to analyze ("all", "train", "test")
        save_report: Generate HTML report
        
    Returns:
        TimeFliesResult: EDA results
        
    Example:
        >>> config = Config(project="alzheimers")  
        >>> eda_results = eda(config, save_report=True)
        >>> # HTML report saved to outputs/
    """
    command = ["eda", "--split", split]
    
    if save_report:
        command.append("--save-report")
    
    return _run_timeflies_command(command, config)


def analyze(config: Config, 
            predictions_path: Optional[str] = None,
            with_eda: bool = False) -> TimeFliesResult:
    """
    Run project-specific analysis.
    
    Args:
        config: TimeFlies configuration
        predictions_path: Path to existing predictions CSV
        with_eda: Run EDA before analysis
        
    Returns:
        TimeFliesResult: Analysis results
    """
    command = ["analyze"]
    
    if predictions_path:
        command.extend(["--predictions-path", predictions_path])
    if with_eda:
        command.append("--with-eda")
    
    return _run_timeflies_command(command, config)


def get_experiment_results(experiment_dir: str) -> Optional[TimeFliesResult]:
    """
    Load results from an existing experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        
    Returns:
        TimeFliesResult: Loaded results or None
    """
    exp_path = Path(experiment_dir)
    if not exp_path.exists():
        return None
    
    # Load metadata
    metadata_file = exp_path / "metadata.json"
    config_data = {}
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            config_data = {
                "project": metadata.get("project", "unknown"),
                "tissue": metadata.get("tissue", "head"), 
                "model": metadata.get("model_type", "cnn").lower(),
                "target": metadata.get("target", "age"),
                "batch_corrected": metadata.get("batch_correction", False),
            }
    
    config = Config(**config_data)
    
    # Load metrics
    metrics = {}
    metrics_file = exp_path / "evaluation" / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
    
    # Load predictions
    predictions = None
    predictions_file = exp_path / "evaluation" / "predictions.csv"
    if predictions_file.exists():
        predictions = pd.read_csv(predictions_file)
    
    return TimeFliesResult(
        config=config,
        success=True,
        experiment_dir=str(exp_path),
        metrics=metrics,
        predictions=predictions
    )


def list_experiments(project: str = "fruitfly_alzheimers") -> List[str]:
    """
    List available experiments for a project.
    
    Args:
        project: Project name
        
    Returns:
        List of experiment directory paths
    """
    experiments = []
    
    for correction in ["uncorrected", "batch_corrected"]:
        exp_base = Path(f"outputs/{project}/experiments/{correction}")
        if exp_base.exists():
            for config_dir in exp_base.iterdir():
                if config_dir.is_dir() and config_dir.name not in ["best", "latest"]:
                    for exp_dir in config_dir.iterdir():
                        if exp_dir.is_dir() and exp_dir.name != "best":
                            experiments.append(str(exp_dir))
    
    return sorted(experiments)