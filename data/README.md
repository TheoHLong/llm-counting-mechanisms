# Data Directory

This directory contains **complete experimental data** from all LLM counting mechanism experiments. All results are provided so you can reproduce analysis and plots without running expensive model evaluations.

## 📁 Directory Structure

### `benchmark_results/` - Model Evaluation Data

#### Core Dataset
- **`counting_dataset_5000.csv`** - Main evaluation dataset (5,000 examples)
  - 11 semantic categories (fruit, animal, vehicle, etc.)
  - Uniform distribution of correct answers (0 to max possible)
  - Variable list lengths (5-10 words)
  - Format: `type, list_items, list_length, answer`

- **`word_banks.json`** - Word categories used for dataset generation
  - 11 categories with 30-80 words each
  - Used for both dataset creation and intervention generation
  - Ensures reproducible random sampling

#### Individual Model Results
- **`llama3_1_8b_results.csv`** - Llama 3.1 8B Instruct detailed results
- **`phi4_results.csv`** - Microsoft Phi-4 detailed results  
- **`qwen3_8b_results.csv`** - Qwen3 8B detailed results

Each file contains:
- `model`: Model name
- `example_id`: Unique example identifier
- `type`: Word category being counted
- `list_items`: Word list presented to model
- `list_length`: Number of words in list
- `expected_answer`: Correct count
- `model_response`: Raw model output
- `extracted_answer`: Parsed numerical answer
- `correct`: Boolean indicating accuracy

#### Comparison and Analysis
- **`model_comparison.csv`** - Cross-model performance summary
  - Overall accuracy by model
  - Total examples and correct counts
  - Used for generating comparison plots

- **`accuracy_comparison_table.csv`** - Detailed accuracy breakdown
  - Performance by category for each model
  - Statistical comparisons

- **`benchmark_report.md`** - Comprehensive analysis report
  - Detailed findings and interpretations
  - Category-wise performance analysis
  - Error pattern identification

### `causal_results/` - Causal Mediation Analysis Data

#### Intervention Data
- **`cma_intervention_pairs.json`** - Counterfactual test cases
  - 3,000+ intervention pairs
  - Each pair: original list → modified list (target word replaced)
  - Format: `original_list, intervention_list, intervention_position, etc.`

#### Effect Calculations
- **`cma_effects_results.csv`** - Complete effect calculations by layer
  - `pair_id`: Intervention pair identifier
  - `layer_idx`: Transformer layer index
  - `TE`: Total Effect (intervention output - original output)
  - `IE`: Indirect Effect (mediated through specific layer)
  - `original_output`, `intervention_output`: Model predictions
  - `original_correct`, `intervention_correct`: Accuracy flags

#### Layer Analysis
- **`cma_layer_statistics.csv`** - Aggregated statistics by layer
  - `layer_idx`: Layer number
  - `mean_TE`, `mean_IE`: Average effect magnitudes
  - `mean_abs_TE`, `mean_abs_IE`: Average absolute effects
  - `count`: Number of calculations per layer

- **`layer_statistics.csv`** - Alternative layer statistics format
  - Compatible with visualization scripts
  - Used for generating layer-wise plots

#### Analysis Reports
- **`cma_analysis_report.txt`** - Detailed causal analysis findings
  - Key layer identifications
  - Effect magnitude summaries
  - Methodological notes

## 🚀 Using the Data

### Quick Visualization
Generate all plots from existing data:

```python
from src.visualization import ResultsVisualizer

# Load data and create all plots
visualizer = ResultsVisualizer("data/")
visualizer.generate_all_plots()
```

### Load Specific Datasets
```python
import pandas as pd

# Load benchmark results
phi4_results = pd.read_csv("data/benchmark_results/phi4_results.csv")
model_comparison = pd.read_csv("data/benchmark_results/model_comparison.csv")

# Load causal analysis results
causal_effects = pd.read_csv("data/causal_results/cma_effects_results.csv")
layer_stats = pd.read_csv("data/causal_results/cma_layer_statistics.csv")

# Load intervention pairs
import json
with open("data/causal_results/cma_intervention_pairs.json", 'r') as f:
    interventions = json.load(f)
```

### Custom Analysis Examples
```python
# Analyze model performance by category
phi4_results = pd.read_csv("data/benchmark_results/phi4_results.csv")
category_performance = phi4_results.groupby('type')['correct'].mean()
print("Phi-4 Performance by Category:")
print(category_performance.sort_values(ascending=False))

# Find peak causal mediation layers
causal_effects = pd.read_csv("data/causal_results/cma_effects_results.csv")
layer_effects = causal_effects.groupby('layer_idx')['IE'].apply(lambda x: abs(x).mean())
peak_layer = layer_effects.idxmax()
print(f"Peak mediation layer: {peak_layer}")
```

## 📊 Data Statistics

### Benchmark Dataset
- **Total Examples**: 5,000
- **Categories**: 11 semantic types
- **Models Evaluated**: 3 (Llama 3.1 8B, Phi-4, Qwen3 8B)
- **Total Predictions**: 15,000 (3 models × 5,000 examples)

### Causal Analysis Dataset  
- **Intervention Pairs**: 3,000+
- **Layers Analyzed**: 32 (Phi-4 architecture)
- **Total Effect Calculations**: 96,000+ (3,000 pairs × 32 layers)
- **Valid Results**: ~85% (after filtering failed calculations)

## 🔬 Research Applications

### Model Comparison Studies
- Cross-model performance analysis
- Category-specific accuracy patterns
- Error analysis and failure modes

### Mechanistic Interpretability
- Layer-wise counting representation analysis
- Causal mediation strength identification
- Distributed processing investigation

### Methodology Development
- Counterfactual intervention techniques
- Activation patching validation
- Effect size measurement methods

## 📝 Data Format Notes

### CSV Files
- All CSV files use standard comma separation
- Headers are included in first row
- Missing values represented as NaN or empty strings

### JSON Files
- Pretty-printed with 2-space indentation
- Unicode characters preserved
- Compatible with standard JSON parsers

### Text Reports
- Plain text format with markdown-style headers
- UTF-8 encoding
- Human-readable analysis summaries

## 🎯 Reproducing Results

All figures in the main README can be reproduced using this data:

1. **Overall Accuracy Comparison**: From `model_comparison.csv`
2. **Category Performance**: From individual model result files
3. **Causal Effect Magnitudes**: From `cma_layer_statistics.csv`
4. **Accuracy Before/After**: From `cma_effects_results.csv`

The visualization module (`src/visualization.py`) automatically detects and uses this data structure.
