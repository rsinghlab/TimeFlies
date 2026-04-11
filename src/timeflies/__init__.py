"""
TimeFlies: snRNA-seq aging clock for Drosophila.

Every CLI command is available as a Python function::

    import timeflies

    timeflies.setup()                        # split data + create dirs
    timeflies.train()                        # train + evaluate + plots
    timeflies.evaluate()                     # evaluate saved model
    timeflies.eda()                          # exploratory data analysis
    timeflies.analyze()                      # project-specific analysis
    timeflies.split()                        # just create data splits
    timeflies.batch_correct()                # apply scVI batch correction
    timeflies.tune("config.yaml")            # hyperparameter tuning
    timeflies.queue("queue.yaml")            # multi-model training
    adata = timeflies.load_data("cells.h5ad")

All functions use configs/default.yaml unless a config path is passed.
"""

from __future__ import annotations

__version__ = "1.0.0"

import scanpy as sc
from anndata import AnnData


def _get_config(config_path: str | None = None):
    """Load config from YAML."""
    from .core.active_config import get_config_for_active_project

    return get_config_for_active_project(config_path)


def load_data(path: str) -> AnnData:
    """Load an h5ad file."""
    return sc.read_h5ad(path)


def setup(config_path: str | None = None, batch_correct: bool = False, force_split: bool = False) -> int:
    """Split data, create directories, verify system. Same as ``timeflies setup``."""
    from .cli.commands.setup import new_setup_command

    class Args:
        pass

    args = Args()
    args.batch_correct = batch_correct
    args.force_split = force_split
    args.tissue = "head"
    args.project = None
    return new_setup_command(args)


def train(config_path: str | None = None) -> dict:
    """Train + evaluate + plots. Same as ``timeflies train``."""
    from .core.pipeline_manager import PipelineManager

    config = _get_config(config_path)
    pipeline = PipelineManager(config, mode="training")
    return pipeline.run_pipeline()


def evaluate(config_path: str | None = None) -> dict:
    """Evaluate a trained model on holdout set. Same as ``timeflies evaluate``."""
    from .core.pipeline_manager import PipelineManager

    config = _get_config(config_path)
    pipeline = PipelineManager(config, mode="evaluation")
    return pipeline.run_evaluation()


def eda(config_path: str | None = None, save_report: bool = False) -> int:
    """Run exploratory data analysis. Same as ``timeflies eda``."""
    from .analysis.eda import EDAHandler

    config = _get_config(config_path)

    class Args:
        pass

    args = Args()
    args.save_report = save_report
    args.tissue = getattr(config.data, "tissue", "head")
    args.batch_corrected = getattr(config.data.batch_correction, "enabled", False)
    args.project = getattr(config, "project", None)

    handler = EDAHandler(config)
    handler.run_eda(args)
    return 0


def analyze(config_path: str | None = None, predictions_path: str | None = None) -> int:
    """Run project-specific analysis. Same as ``timeflies analyze``."""
    from .cli.commands.analysis import analyze_command

    config = _get_config(config_path)

    class Args:
        pass

    args = Args()
    args.predictions_path = predictions_path
    args.with_eda = False
    args.analysis_script = None
    args.tissue = getattr(config.data, "tissue", "head")
    args.batch_corrected = getattr(config.data.batch_correction, "enabled", False)
    args.project = getattr(config, "project", None)
    args.verbose = False
    return analyze_command(args, config)


def split(config_path: str | None = None, force: bool = False) -> int:
    """Create train/eval data splits. Same as ``timeflies split``."""
    from .cli.commands.setup import split_command

    class Args:
        pass

    args = Args()
    args.force_split = force
    args.tissue = "head"
    args.project = None
    args.batch_corrected = False
    return split_command(args)


def batch_correct(config_path: str | None = None) -> int:
    """Apply scVI batch correction. Same as ``timeflies batch-correct``."""
    from .cli.commands.training import batch_command

    config = _get_config(config_path)

    class Args:
        pass

    args = Args()
    args.tissue = getattr(config.data, "tissue", "head")
    args.project = getattr(config, "project", None)
    args.verbose = False
    return batch_command(args)


def tune(config_path: str = "examples/hyperparameter_tuning.yaml", resume: bool = True) -> int:
    """Run hyperparameter tuning. Same as ``timeflies tune``."""
    from .cli.commands.advanced import tune_command

    class Args:
        pass

    args = Args()
    args.config = config_path
    args.no_resume = not resume
    return tune_command(args)


def queue(config_path: str = "examples/model_queue.yaml", resume: bool = True) -> int:
    """Run multi-model training queue. Same as ``timeflies queue``."""
    from .cli.commands.advanced import queue_command

    class Args:
        pass

    args = Args()
    args.config = config_path
    args.no_resume = not resume
    return queue_command(args)


__all__ = [
    "load_data",
    "setup",
    "train",
    "evaluate",
    "eda",
    "analyze",
    "split",
    "batch_correct",
    "tune",
    "queue",
]
