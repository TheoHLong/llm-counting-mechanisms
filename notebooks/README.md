# Notebooks Directory

This directory contains the **original Google Colab notebooks** used for the main experiments in the LLM Counting Mechanisms research.

## 📚 Notebook Overview

### 1. `01_data_generation.ipynb`
- **Purpose**: Generate the counting task dataset with 11 semantic categories
- **Output**: 
  - `counting_dataset_5000.csv` - Main evaluation dataset
  - `word_banks.json` - Word categories for reproducibility
  - `intervention_pairs.json` - Pairs for causal analysis
- **Features**:
  - Comprehensive word banks (11 categories, 30-80 words each)
  - Uniform distribution of correct answers
  - Variable list lengths (5-10 words)
  - Dataset validation and statistics

### 2. `02_model_benchmark.ipynb`
- **Purpose**: Zero-shot evaluation of multiple LLMs on counting tasks
- **Models Tested**:
  - Llama 3.1 8B Instruct
  - Microsoft Phi-4
  - Qwen3 8B
- **Key Features**:
  - **No chat templates** used for fair cross-model comparison
  - Modular model evaluation framework
  - Comprehensive accuracy analysis by category
  - Publication-quality visualization generation
- **Output**: Model performance comparisons and accuracy plots

### 3. `03_causal_analysis.ipynb`
- **Purpose**: Causal mediation analysis using counterfactual activation patching
- **Model**: Microsoft Phi-4 **with chat templates**
- **Method**: 
  - Counterfactual activation patching
  - Total Effect (TE) and Indirect Effect (IE) calculation
  - Layer-by-layer analysis of counting representations
- **Key Findings**: 
  - Layers 15-18 show strongest mediation effects
  - Evidence for distributed counting mechanisms
  - Layer 16 identified as peak "counting layer"

## 🚀 Running the Notebooks

### Execution Order
For complete reproduction, run the notebooks in numerical order:

1. **First**: `01_data_generation.ipynb` - Creates the dataset and word banks
2. **Second**: `02_model_benchmark.ipynb` - Evaluates models on the dataset  
3. **Third**: `03_causal_analysis.ipynb` - Performs causal mediation analysis

### Google Colab (Recommended)
1. Upload notebooks to Google Colab
2. Mount Google Drive for data persistence:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Install required packages:
   ```python
   !pip install transformers torch pandas matplotlib seaborn tqdm
   ```
4. Run cells sequentially

### Local Jupyter
1. Install Jupyter and dependencies:
   ```bash
   pip install jupyter transformers torch pandas matplotlib seaborn tqdm
   ```
2. Start Jupyter:
   ```bash
   jupyter notebook
   ```
3. Open and run notebooks

## 📊 Expected Outputs

### Data Generation
- Dataset files in `/content/counting_dataset/` (Colab) or `../data/` (local)
- Validation statistics and sample examples
- Word bank summaries

### Model Benchmark
- Individual model result CSVs
- Overall accuracy comparison plots
- Category-wise performance breakdowns
- Model comparison summary

### Causal Analysis
- Effect calculation results (TE/IE by layer)
- Layer-wise mediation strength plots
- Accuracy before/after intervention comparisons
- Summary statistics of causal effects

## 🔧 Modifications for Local Use

If running locally instead of Colab, update file paths:

```python
# Change from Colab paths:
# output_dir = "/content/counting_dataset/"

# To local paths:
output_dir = "../data/"
results_dir = "../results/"
```

## 💡 Key Experimental Decisions

### Why Different Chat Template Usage?
- **Benchmark**: No chat templates for fair cross-model comparison
- **Causal Analysis**: Chat templates with Phi-4 for realistic intervention analysis
- This methodological difference is intentional and scientifically justified

### Why Google Colab?
- **GPU Access**: Free T4/V100 GPUs for model inference
- **Reproducibility**: Consistent environment across runs
- **Collaboration**: Easy sharing and version control
- **Storage**: Integration with Google Drive for large files

## 📈 Research Impact

These notebooks represent the core experimental methodology that led to:
- First systematic comparison of LLM counting abilities across model families
- Novel application of causal mediation analysis to numerical reasoning
- Identification of specific transformer layers responsible for counting behavior
- Evidence for distributed counting mechanisms in large language models

## 🔬 For Researchers

The notebooks provide:
- **Complete experimental protocols** for reproduction
- **Raw experimental code** with all implementation details
- **Intermediate outputs** and debugging information
- **Parameter settings** and hyperparameter choices
- **Data processing pipelines** from generation to analysis

Use these notebooks as the authoritative reference for reproducing the main experimental results.
