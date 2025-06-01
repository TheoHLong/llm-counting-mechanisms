#!/usr/bin/env python3
"""
Complete benchmark pipeline for LLM counting task evaluation.

This script runs the full benchmark including:
1. Dataset generation (if needed)
2. Model evaluation
3. Results visualization

Usage:
    python scripts/run_benchmark.py --models microsoft/phi-4 meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_generation import CountingDataGenerator
from model_benchmark import CountingBenchmark
from visualization import ResultsVisualizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run complete LLM counting benchmark")
    parser.add_argument("--models", nargs="+", required=True, 
                       help="Model names to evaluate")
    parser.add_argument("--dataset-size", type=int, default=1000,
                       help="Number of examples to evaluate (default: 1000)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--hf-token", type=str,
                       help="HuggingFace token for gated models")
    parser.add_argument("--regenerate-data", action="store_true",
                       help="Regenerate dataset even if it exists")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup paths
    dataset_path = os.path.join(args.data_dir, "counting_dataset_5000.csv")
    word_banks_path = os.path.join(args.data_dir, "word_banks.json")
    
    # Step 1: Generate dataset if needed
    if not os.path.exists(dataset_path) or args.regenerate_data:
        logger.info("Generating counting dataset...")
        generator = CountingDataGenerator()
        dataset = generator.create_dataset(size=5000)
        generator.save_dataset(dataset, dataset_path)
        generator.save_word_banks(word_banks_path)
        logger.info(f"Dataset saved to {dataset_path}")
    else:
        logger.info(f"Using existing dataset: {dataset_path}")
    
    # Step 2: Setup tokens for gated models
    tokens = {}
    if args.hf_token:
        for model in args.models:
            if "llama" in model.lower() or "meta" in model.lower():
                tokens[model] = args.hf_token
    
    # Step 3: Run benchmark
    logger.info(f"Starting benchmark with models: {args.models}")
    benchmark = CountingBenchmark(dataset_path, args.output_dir)
    
    try:
        results = benchmark.evaluate_models(
            args.models, 
            num_examples=args.dataset_size,
            tokens=tokens
        )
        
        logger.info("Benchmark completed successfully!")
        
        # Step 4: Generate visualizations
        logger.info("Generating visualizations...")
        visualizer = ResultsVisualizer(args.output_dir)
        
        # Load results and create plots
        benchmark_results = visualizer.load_benchmark_results()
        if benchmark_results:
            visualizer.plot_overall_accuracy(benchmark_results)
            visualizer.plot_category_accuracy(benchmark_results)
            logger.info("Visualizations saved to results/figures/")
        
        # Step 5: Print summary
        print("\n" + "="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        for model_name, df in results.items():
            accuracy = df['correct'].mean()
            print(f"{model_name:25s}: {accuracy:.2%} ({df['correct'].sum()}/{len(df)})")
        
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Visualizations in: {os.path.join(args.output_dir, 'figures')}")
        
    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


if __name__ == "__main__":
    main()
