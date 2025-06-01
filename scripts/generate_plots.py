#!/usr/bin/env python3
"""
Generate all visualizations for existing results.

This script creates plots and summary reports for any existing
benchmark or causal analysis results.

Usage:
    python scripts/generate_plots.py
    python scripts/generate_plots.py --results-dir custom_results/
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from visualization import ResultsVisualizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate all visualizations")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory to process")
    
    args = parser.parse_args()
    
    logger.info(f"Generating visualizations for results in: {args.results_dir}")
    
    try:
        visualizer = ResultsVisualizer(args.results_dir)
        visualizer.generate_all_plots()
        
        print("\n" + "="*60)
        print("VISUALIZATION GENERATION COMPLETE")
        print("="*60)
        print(f"Plots saved to: {Path(args.results_dir) / 'figures'}")
        print(f"Summary report: {Path(args.results_dir) / 'analysis_summary.md'}")
        
    except Exception as e:
        logger.error(f"Visualization generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
