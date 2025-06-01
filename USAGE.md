# Quick Start Examples

This document provides quick examples for using the LLM Counting Mechanisms toolkit.

## 🚀 Basic Usage

### 1. Generate Dataset
```python
from src.data_generation import CountingDataGenerator

# Create generator
generator = CountingDataGenerator()

# Generate 5000 examples
dataset = generator.create_dataset(size=5000)

# Save to file
generator.save_dataset(dataset, "data/counting_dataset_5000.csv")
generator.save_word_banks("data/word_banks.json")

# Show statistics
stats = generator.validate_dataset(dataset)
print(f"Generated {stats['total_examples']} examples")
```

### 2. Benchmark Models
```python
from src.model_benchmark import CountingBenchmark

# Initialize benchmark
benchmark = CountingBenchmark("data/counting_dataset_5000.csv")

# Evaluate single model
results = benchmark.evaluate_model("microsoft/phi-4", num_examples=1000)
print(f"Accuracy: {results['correct'].mean():.2%}")

# Evaluate multiple models
models = ["microsoft/phi-4", "Qwen/Qwen3-8B"]
all_results = benchmark.evaluate_models(models, num_examples=1000)
```

### 3. Causal Analysis
```python
from src.causal_analysis import CausalMediationAnalyzer, InterventionDataGenerator

# Generate intervention pairs
generator = InterventionDataGenerator("data/word_banks.json")
pairs = generator.generate_intervention_dataset(
    num_pairs=1000, 
    output_path="data/intervention_pairs.json"
)

# Run causal analysis
analyzer = CausalMediationAnalyzer("microsoft/phi-4")
effects = analyzer.run_analysis(
    pairs_path="data/intervention_pairs.json",
    max_pairs=100
)

# Examine results
valid_effects = effects[effects['TE'].notna()]
print(f"Mean |IE|: {abs(valid_effects['IE']).mean():.3f}")
```

### 4. Generate Visualizations
```python
from src.visualization import ResultsVisualizer

# Create visualizer
visualizer = ResultsVisualizer("results/")

# Load and plot benchmark results
benchmark_results = visualizer.load_benchmark_results()
visualizer.plot_overall_accuracy(benchmark_results)
visualizer.plot_category_accuracy(benchmark_results)

# Load and plot causal results
causal_results = visualizer.load_causal_results()
if causal_results is not None:
    visualizer.plot_causal_effects(causal_results)
    visualizer.plot_accuracy_comparison(causal_results)

# Generate all plots at once
visualizer.generate_all_plots()
```

## 🖥️ Command Line Usage

### Run Complete Benchmark
```bash
# Basic benchmark with 2 models
python scripts/run_benchmark.py \
    --models microsoft/phi-4 Qwen/Qwen3-8B \
    --dataset-size 1000 \
    --output-dir results

# With HuggingFace token for gated models
python scripts/run_benchmark.py \
    --models meta-llama/Meta-Llama-3.1-8B-Instruct microsoft/phi-4 \
    --hf-token your_hf_token_here \
    --dataset-size 1000
```

### Run Causal Analysis
```bash
# Basic causal analysis
python scripts/run_causal_analysis.py \
    --model microsoft/phi-4 \
    --max-pairs 100 \
    --output-dir results

# Test specific layers
python scripts/run_causal_analysis.py \
    --model microsoft/phi-4 \
    --layers 10 11 12 13 14 15 16 17 18 19 \
    --max-pairs 200
```

### Generate All Plots
```bash
# Generate visualizations for existing results
python scripts/generate_plots.py --results-dir results
```

## 📊 Understanding Results

### Model Performance
- **Overall Accuracy**: Percentage of correctly counted examples
- **Category Performance**: Accuracy breakdown by word type (fruit, animal, etc.)
- **Error Analysis**: Common failure patterns

### Causal Effects
- **Total Effect (TE)**: Change in output when input is modified
- **Indirect Effect (IE)**: Effect mediated through specific layer
- **Layer Analysis**: Which layers contain counting representations

### Key Findings
- **Peak Layers**: Layers 15-18 typically show strongest effects
- **Distributed Processing**: Multiple layers contribute to counting
- **Model Differences**: Performance varies significantly across models

## 🔧 Customization

### Adding New Models
```python
from src.model_benchmark import ModelEvaluator

class MyModelEvaluator(ModelEvaluator):
    def load_model(self):
        # Your model loading code
        pass
    
    def generate_response(self, messages):
        # Your inference code
        pass
```

### Custom Datasets
```python
# Modify word banks
generator = CountingDataGenerator()
generator.word_banks["my_category"] = ["word1", "word2", "word3"]

# Generate with custom parameters
dataset = generator.create_dataset(size=1000)
```

### Custom Analysis
```python
# Analyze specific layers
analyzer = CausalMediationAnalyzer("microsoft/phi-4")
results = analyzer.run_analysis(
    pairs_path="data/intervention_pairs.json",
    layers_to_test=[15, 16, 17, 18],  # Focus on key layers
    max_pairs=500
)
```

## 📈 Interpreting Visualizations

### Overall Accuracy Plot
- Shows comparative performance across models
- Higher bars = better counting ability
- Use for model selection

### Category Accuracy Plot
- Shows which word types are harder/easier
- Grouped bars compare models on each category
- Reveals model-specific strengths/weaknesses

### Causal Effects Plot
- Shows which layers mediate counting
- Peak in middle-to-late layers indicates counting representation
- Multiple peaks suggest distributed processing

### Accuracy Comparison Plot
- Shows impact of interventions
- Large drops indicate successful causal manipulation
- Validates that interventions affect relevant computations

## 🎯 Research Applications

### Mechanistic Interpretability
- Identify where counting happens in transformer layers
- Compare counting mechanisms across model families
- Study emergence of numerical reasoning

### Model Evaluation
- Benchmark numerical reasoning capabilities
- Identify failure modes and edge cases
- Guide model development and improvement

### Educational Use
- Demonstrate causal analysis techniques
- Teach activation patching methods
- Illustrate mechanistic interpretability concepts

## 🚨 Common Issues

### Memory Problems
- Use smaller batch sizes for large models
- Process fewer examples at once
- Enable gradient checkpointing if available

### Model Loading Errors
- Check HuggingFace token for gated models
- Verify model names are correct
- Ensure sufficient disk space

### Visualization Issues
- Check that result files exist in the correct directory
- Verify matplotlib backend compatibility
- Ensure all dependencies are installed

## 📝 Tips for Success

1. **Start Small**: Begin with 100-500 examples for testing
2. **Monitor Resources**: Track GPU memory usage during analysis
3. **Save Intermediate Results**: Use checkpointing for long runs
4. **Validate Results**: Manually check a few examples
5. **Document Experiments**: Keep track of parameter settings

## 🔗 Next Steps

- Extend to other numerical reasoning tasks
- Explore different intervention strategies
- Analyze cross-linguistic counting abilities
- Study few-shot vs zero-shot performance
- Investigate chain-of-thought reasoning effects
