"""
LLM Counting Mechanisms: Behavioral Analysis and Causal Mediation

A comprehensive investigation into how Large Language Models process counting tasks.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .data_generation import CountingDataGenerator
from .model_benchmark import CountingBenchmark
from .causal_analysis import CausalMediationAnalyzer, InterventionDataGenerator
from .visualization import ResultsVisualizer

__all__ = [
    "CountingDataGenerator",
    "CountingBenchmark", 
    "CausalMediationAnalyzer",
    "InterventionDataGenerator",
    "ResultsVisualizer"
]
