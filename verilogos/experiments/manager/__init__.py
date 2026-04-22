"""
Experiment Manager Module

Core experiment management infrastructure for VeriLogos.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from verilogos.experiments.manager.tracker import ExperimentTracker
from verilogos.experiments.manager.parallel_executor import ParallelExecutor, BatchExecutor
from verilogos.experiments.manager.ablation import AblationGenerator
from verilogos.experiments.manager.plotting import ExperimentPlotter

__all__ = [
    'ExperimentTracker',
    'ParallelExecutor',
    'BatchExecutor',
    'AblationGenerator',
    'ExperimentPlotter',
]
