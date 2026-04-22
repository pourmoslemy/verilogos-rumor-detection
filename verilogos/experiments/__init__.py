"""
VeriLogos Experiments Module

NeurIPS-grade experiment management system.

Author: Alireza Pourmoslemi
Email: apmath99@gmail.com
"""

from verilogos.experiments.manager import (
    ExperimentTracker,
    ParallelExecutor,
    AblationGenerator,
    ExperimentPlotter
)

from verilogos.experiments.manager.experiment_manager import ExperimentManager

__version__ = "1.0.0"
__author__ = "Alireza Pourmoslemi"
__email__ = "apmath99@gmail.com"

__all__ = [
    'ExperimentTracker',
    'ParallelExecutor',
    'AblationGenerator',
    'ExperimentPlotter',
    'ExperimentManager',
]
