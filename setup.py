#!/usr/bin/env python3
"""
Setup script for TimeFlies installation.
Ensures proper package and script installation.
"""

from pathlib import Path

import tomli
from setuptools import find_packages, setup

# Read the contents of README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements from pyproject.toml
with open("pyproject.toml", "rb") as f:
    pyproject = tomli.load(f)

setup(
    name="timeflies",
    version="1.0.0",
    description="Machine Learning for Aging Analysis in Drosophila Single-Cell Data with Deep Learning and Batch Correction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Singh Lab",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=pyproject["project"]["dependencies"],
    extras_require=pyproject["project"]["optional-dependencies"],
    entry_points={
        "console_scripts": [
            "timeflies=common.timeflies_cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
