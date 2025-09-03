"""
Benchmarking Module for CausalPriorFitting.

This module provides tools for benchmarking models on standard datasets.
"""

from .load_openml_benchmark import SimpleOpenMLLoader, DEFAULT_TABULAR_NUM_REG_TASKS

__all__ = ["SimpleOpenMLLoader", "DEFAULT_TABULAR_NUM_REG_TASKS"]
