#!/usr/bin/env python3
"""
Causal mediation analysis pipeline for LLM counting tasks.

This script runs the complete causal analysis including:
1. Intervention pair generation (if needed)
2. Causal mediation analysis
3. Results visualization

Usage:
    python scripts/run_causal_analysis.py --model microsoft/phi-4 --max-pairs 100
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from causal_analysis import CausalMediationAnalyzer, InterventionDataGenerator
from visualization import ResultsVisualizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run causal mediation analysis")
    parser.add_argument("--model", type=str, default="microsoft/phi-4",
                       help="Model name for analysis")
    parser.add_argument("--max-pairs", type=int, default=100,
                       help="Maximum number of intervention pairs to process")
    parser.add_argument("--num-pairs", type=int, default=1000,
                       help="Number of intervention pairs to generate")
    parser.add_argument("--layers", nargs="+", type=int,
                       help="Specific layers to test (default: middle to end layers)")
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Data directory")
    parser.add_argument("--regenerate-pairs", action="store_true",
                       help="Regenerate intervention pairs even if they exist")
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup paths
    word_banks_path = os.path.join(args.data_dir, "word_banks.json")
    pairs_path = os.path.join(args.data_dir, "intervention_pairs.json")
    
    # Step 1: Check if word banks exist
    if not os.path.exists(word_banks_path):
        logger.error(f"Word banks not found at {word_banks_path}")
        logger.error("Please run the benchmark script first to generate the dataset")
        return
    
    # Step 2: Generate intervention pairs if needed
    if not os.path.exists(pairs_path) or args.regenerate_pairs:
        logger.info("Generating intervention pairs...")
        generator = InterventionDataGenerator(word_banks_path)
        pairs = generator.generate_intervention_dataset(
            num_pairs=args.num_pairs,
            output_path=pairs_path
        )
        logger.info(f"Generated {len(pairs)} intervention pairs")
    else:
        logger.info(f"Using existing intervention pairs: {pairs_path}")
    
    # Step 3: Run causal mediation analysis
    logger.info(f"Starting causal mediation analysis with {args.model}")
    
    try:
        analyzer = CausalMediationAnalyzer(args.model)
        
        results = analyzer.run_analysis(
            pairs_path=pairs_path,
            layers_to_test=args.layers,
            max_pairs=args.max_pairs,
            output_dir=args.output_dir
        )
        
        logger.info("Causal mediation analysis completed successfully!")
        
        # Step 4: Generate visualizations
        logger.info("Generating causal analysis visualizations...")
        visualizer = ResultsVisualizer(args.output_dir)
        
        # Load causal results and create plots
        causal_results = visualizer.load_causal_results()
        if causal_results is not None:
            visualizer.plot_causal_effects(causal_results)
            visualizer.plot_accuracy_comparison(causal_results)
            logger.info("Causal visualizations saved to results/figures/")
        
        # Step 5: Print summary
        print("\n" + "="*60)
        print("CAUSAL MEDIATION ANALYSIS SUMMARY")
        print("="*60)
        
        if len(results) > 0:
            valid_results = results[results['TE'].notna()]
            
            print(f"Total calculations: {len(results)}")
            print(f"Valid results: {len(valid_results)}")
            
            if len(valid_results) > 0:
                print(f"Mean |TE|: {abs(valid_results['TE']).mean():.3f}")
                print(f"Mean |IE|: {abs(valid_results['IE']).mean():.3f}")
                
                # Find top layers by indirect effect
                layer_effects = valid_results.groupby('layer_idx')['IE'].apply(
                    lambda x: abs(x).mean()
                ).sort_values(ascending=False)
                
                print(f"\nTop 5 Layers by |IE|:")
                for layer, effect in layer_effects.head().items():
                    print(f"  Layer {layer}: {effect:.3f}")
        
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Visualizations in: {os.path.join(args.output_dir, 'figures')}")
        
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Causal analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
