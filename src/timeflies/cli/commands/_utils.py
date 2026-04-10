"""
Shared utilities for TimeFlies CLI commands.

Contains stderr suppression context manager and conditional batch correction imports.
"""

import contextlib
import os
import sys


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
