"""
Visualization utilities for LLM counting task analysis.

This module provides functions to create publication-quality plots for
both behavioral benchmarking and causal mediation analysis results.

Example usage:
    visualizer = ResultsVisualizer("results/")
    visualizer.plot_benchmark_comparison()
    visualizer.plot_causal_effects()
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ResultsVisualizer:
    """Visualization utilities for counting task results."""
    
    def __init__(self, results_dir: str = "results"):
        """Initialize with results directory."""
        self.results_dir = results_dir
        self.figures_dir = os.path.join(results_dir, "figures")
        os.makedirs(self.figures_dir, exist_ok=True)
    
    def load_benchmark_results(self) -> Dict[str, pd.DataFrame]:
        """Load benchmark results from CSV files."""
        results = {}
        
        # Check both results/ and data/benchmark_results/ directories
        search_dirs = [
            self.results_dir,
            os.path.join(self.results_dir, "benchmark_results"),
            "data/benchmark_results"
        ]
        
        for search_dir in search_dirs:
            if not os.path.exists(search_dir):
                continue
                
            # Look for result files
            for filename in os.listdir(search_dir):
                if filename.endswith("_results.csv"):
                    # Extract model name from filename
                    if "llama3_1_8b" in filename:
                        model_name = "Llama 3.1 8B"
                    elif "phi4" in filename:
                        model_name = "Phi-4"
                    elif "qwen3_8b" in filename:
                        model_name = "Qwen3 8B"
                    else:
                        model_name = filename.replace("_results.csv", "").replace("_", " ").title()
                    
                    filepath = os.path.join(search_dir, filename)
                    
                    try:
                        df = pd.read_csv(filepath)
                        results[model_name] = df
                        logger.info(f"Loaded results for {model_name}: {len(df)} examples")
                    except Exception as e:
                        logger.error(f"Error loading {filename}: {e}")
        
        return results
    
    def load_causal_results(self) -> Optional[pd.DataFrame]:
        """Load causal mediation analysis results."""
        # Check multiple possible locations
        causal_paths = [
            os.path.join(self.results_dir, "causal_effects_results.csv"),
            os.path.join(self.results_dir, "causal_results", "cma_effects_results.csv"),
            "data/causal_results/cma_effects_results.csv"
        ]
        
        for causal_path in causal_paths:
            if os.path.exists(causal_path):
                try:
                    df = pd.read_csv(causal_path)
                    logger.info(f"Loaded causal results: {len(df)} effect calculations")
                    return df
                except Exception as e:
                    logger.error(f"Error loading causal results from {causal_path}: {e}")
        
        logger.warning("No causal results file found in any expected location")
        return None
    
    def plot_overall_accuracy(self, results: Dict[str, pd.DataFrame], 
                            save_path: Optional[str] = None) -> None:
        """Plot overall accuracy comparison across models."""
        
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "overall_accuracy_comparison.png")
        
        # Calculate accuracies
        models = []
        accuracies = []
        
        for model_name, df in results.items():
            accuracy = df['correct'].mean() * 100
            models.append(model_name)
            accuracies.append(accuracy)
        
        # Sort by accuracy
        sorted_data = sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)
        models, accuracies = zip(*sorted_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax.bar(models, accuracies, color=colors[:len(models)], alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom', 
                   fontsize=12, fontweight='bold')
        
        # Customize plot
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_title('LLM Performance on Counting Task (Zero-shot)', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim(0, max(accuracies) * 1.15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.xticks(fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved overall accuracy plot to {save_path}")
    
    def plot_category_accuracy(self, results: Dict[str, pd.DataFrame],
                             save_path: Optional[str] = None) -> None:
        """Plot accuracy by word category for each model."""
        
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "type_accuracy_comparison.png")
        
        # Calculate category accuracies
        category_data = {}
        all_categories = set()
        
        for model_name, df in results.items():
            category_acc = df.groupby('type')['correct'].mean() * 100
            category_data[model_name] = category_acc
            all_categories.update(category_acc.index)
        
        all_categories = sorted(all_categories)
        
        # Prepare data for plotting
        models = list(results.keys())
        x = np.arange(len(all_categories))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, model in enumerate(models):
            values = [category_data[model].get(cat, 0) for cat in all_categories]
            positions = x + (i - len(models)/2 + 0.5) * width
            
            bars = ax.bar(positions, values, width, label=model, 
                         color=colors[i % len(colors)], alpha=0.8, 
                         edgecolor='black', linewidth=1)
            
            # Add value labels for bars > 50%
            for bar, val in zip(bars, values):
                if val > 50:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Customize plot
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Word Type', fontsize=14, fontweight='bold')
        ax.set_title('LLM Performance by Word Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(all_categories, fontsize=12, rotation=45, ha='right')
        ax.legend(fontsize=12, loc='upper left')
        ax.set_ylim(0, 105)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved category accuracy plot to {save_path}")
    
    def plot_causal_effects(self, causal_df: pd.DataFrame,
                          save_path: Optional[str] = None) -> None:
        """Plot causal effect magnitudes by layer."""
        
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "effect_magnitudes_by_layer.png")
        
        # Filter valid results
        valid_df = causal_df[causal_df['TE'].notna()]
        
        if len(valid_df) == 0:
            logger.warning("No valid causal results to plot")
            return
        
        # Calculate layer statistics
        layer_stats = valid_df.groupby('layer_idx').agg({
            'TE': lambda x: np.abs(x).mean(),
            'IE': lambda x: np.abs(x).mean()
        }).reset_index()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = layer_stats['layer_idx']
        plt.plot(x, layer_stats['TE'], 'o-', label='|TE| (Total Effect)',
                markersize=8, linewidth=2, color='blue')
        plt.plot(x, layer_stats['IE'], '^-', label='|IE| (Indirect Effect)',
                markersize=8, linewidth=2, color='red')
        
        plt.xlabel('Layer Index', fontsize=14, fontweight='bold')
        plt.ylabel('Mean Absolute Effect', fontsize=14, fontweight='bold')
        plt.title('Causal Effect Magnitudes by Layer', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Ensure integer x-axis ticks
        plt.xticks(x.astype(int))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved causal effects plot to {save_path}")
    
    def plot_accuracy_comparison(self, causal_df: pd.DataFrame,
                               save_path: Optional[str] = None) -> None:
        """Plot accuracy before vs after interventions."""
        
        if save_path is None:
            save_path = os.path.join(self.figures_dir, "overall_correct_rates.png")
        
        # Filter valid results
        valid_df = causal_df[causal_df['TE'].notna()]
        
        if len(valid_df) == 0:
            logger.warning("No valid causal results to plot")
            return
        
        # Calculate accuracy rates
        original_rate = valid_df['original_correct'].mean()
        intervention_rate = valid_df['intervention_correct'].mean()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Original', 'After Intervention']
        rates = [original_rate, intervention_rate]
        colors = ['skyblue', 'lightcoral']
        
        bars = ax.bar(categories, rates, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                   f'{rate:.2%}', ha='center', va='bottom', 
                   fontsize=14, fontweight='bold')
        
        ax.set_ylabel('Correct Rate', fontsize=14, fontweight='bold')
        ax.set_title('Model Accuracy: Original vs After Intervention', 
                    fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.yaxis.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved accuracy comparison plot to {save_path}")
    
    def create_summary_report(self, benchmark_results: Optional[Dict[str, pd.DataFrame]] = None,
                            causal_results: Optional[pd.DataFrame] = None) -> str:
        """Create a text summary report of all results."""
        
        report_lines = []
        report_lines.append("# LLM Counting Mechanisms - Results Summary")
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Benchmark results
        if benchmark_results:
            report_lines.append("## Model Performance Benchmark")
            report_lines.append("")
            
            for model_name, df in benchmark_results.items():
                accuracy = df['correct'].mean()
                total = len(df)
                correct = df['correct'].sum()
                
                report_lines.append(f"### {model_name}")
                report_lines.append(f"- Overall Accuracy: {accuracy:.2%} ({correct}/{total})")
                
                # Category breakdown
                category_acc = df.groupby('type')['correct'].mean().sort_values(ascending=False)
                best_category = category_acc.index[0]
                worst_category = category_acc.index[-1]
                
                report_lines.append(f"- Best Category: {best_category} ({category_acc[best_category]:.2%})")
                report_lines.append(f"- Worst Category: {worst_category} ({category_acc[worst_category]:.2%})")
                report_lines.append("")
        
        # Causal analysis results
        if causal_results is not None:
            valid_causal = causal_results[causal_results['TE'].notna()]
            
            if len(valid_causal) > 0:
                report_lines.append("## Causal Mediation Analysis")
                report_lines.append("")
                
                # Overall statistics
                mean_te = np.abs(valid_causal['TE']).mean()
                mean_ie = np.abs(valid_causal['IE']).mean()
                original_acc = valid_causal['original_correct'].mean()
                intervention_acc = valid_causal['intervention_correct'].mean()
                
                report_lines.append(f"- Total Calculations: {len(causal_results)}")
                report_lines.append(f"- Valid Results: {len(valid_causal)}")
                report_lines.append(f"- Mean |Total Effect|: {mean_te:.3f}")
                report_lines.append(f"- Mean |Indirect Effect|: {mean_ie:.3f}")
                report_lines.append(f"- Original Accuracy: {original_acc:.2%}")
                report_lines.append(f"- Post-Intervention Accuracy: {intervention_acc:.2%}")
                report_lines.append("")
                
                # Top layers by indirect effect
                layer_effects = valid_causal.groupby('layer_idx')['IE'].apply(
                    lambda x: np.abs(x).mean()
                ).sort_values(ascending=False)
                
                report_lines.append("### Top 5 Layers by Indirect Effect Magnitude")
                for i, (layer, effect) in enumerate(layer_effects.head().items()):
                    report_lines.append(f"{i+1}. Layer {layer}: {effect:.3f}")
                
                report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report
        report_path = os.path.join(self.results_dir, "analysis_summary.md")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Saved summary report to {report_path}")
        
        return report_text
    
    def generate_all_plots(self) -> None:
        """Generate all available plots based on existing data."""
        
        logger.info("Generating all available visualizations...")
        
        # Load benchmark results
        benchmark_results = self.load_benchmark_results()
        if benchmark_results:
            self.plot_overall_accuracy(benchmark_results)
            self.plot_category_accuracy(benchmark_results)
        
        # Load causal results
        causal_results = self.load_causal_results()
        if causal_results is not None:
            self.plot_causal_effects(causal_results)
            self.plot_accuracy_comparison(causal_results)
        
        # Generate summary report
        self.create_summary_report(benchmark_results, causal_results)
        
        logger.info("All visualizations generated successfully!")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate visualizations for counting task results")
    parser.add_argument("--results-dir", type=str, default="results",
                       help="Results directory to process")
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.results_dir)
    visualizer.generate_all_plots()
