"""
Shared utilities for TimeFlies CLI commands.

Contains environment variable suppression, stderr suppression context manager,
and conditional batch correction imports.
"""

import contextlib
import os
import sys

# Suppress TensorFlow logging before any imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr


# Import batch correction class (dependencies checked at instantiation)
try:
    from ...data.preprocessing.batch_correction import BatchCorrector

    BATCH_CORRECTION_AVAILABLE = True
except ImportError:
    BATCH_CORRECTION_AVAILABLE = False
    BatchCorrector = None
