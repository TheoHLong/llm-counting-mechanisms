"""
LLM Counting Task Benchmark

This module provides a comprehensive framework for benchmarking multiple
Large Language Models on word counting tasks using zero-shot evaluation.

Example usage:
    benchmark = CountingBenchmark("data/counting_dataset.csv")
    results = benchmark.evaluate_models(["microsoft/phi-4", "meta-llama/Llama-2-7b-hf"])
    benchmark.generate_comparison_plots(results)
"""

import transformers
import torch
import csv
import re
import pandas as pd
from tqdm import tqdm
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import os
import warnings
import gc
from datetime import datetime

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator(ABC):
    """Abstract base class for model evaluators."""
    
    def __init__(self, model_name: str, token: Optional[str] = None):
        self.model_name = model_name
        self.token = token
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.pipeline = None
    
    @abstractmethod
    def load_model(self, **kwargs) -> None:
        """Load the model and tokenizer/processor."""
        pass
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from the model."""
        pass
    
    def cleanup(self) -> None:
        """Clean up model from memory."""
        # Delete model components
        for attr in ['model', 'tokenizer', 'processor', 'pipeline']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        
        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        logger.info(f"Cleaned up {self.model_name} from memory")


class LlamaEvaluator(ModelEvaluator):
    """Evaluator for Llama models."""
    
    def load_model(self, **kwargs) -> None:
        logger.info(f"Loading {self.model_name}...")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
            token=self.token,
        )
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        outputs = self.pipeline(
            messages,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][-1]["content"]


class PhiEvaluator(ModelEvaluator):
    """Evaluator for Phi models."""
    
    def load_model(self, **kwargs) -> None:
        logger.info(f"Loading {self.model_name}...")
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_name,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        outputs = self.pipeline(
            messages,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )
        return outputs[0]["generated_text"][-1]["content"]


class QwenEvaluator(ModelEvaluator):
    """Evaluator for Qwen models."""
    
    def load_model(self, **kwargs) -> None:
        logger.info(f"Loading {self.model_name}...")
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
    
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")


class CountingBenchmark:
    """Main benchmark class for evaluating models on counting tasks."""
    
    def __init__(self, dataset_path: str, output_dir: str = "results"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.dataset = self._load_dataset()
        
        # Model mapping
        self.model_evaluators = {
            "llama": LlamaEvaluator,
            "phi": PhiEvaluator,
            "qwen": QwenEvaluator,
        }
    
    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load the counting dataset from CSV file."""
        logger.info(f"Loading dataset from {self.dataset_path}")
        data = []
        
        with open(self.dataset_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse list items from string format
                list_items = eval(row['list_items']) if isinstance(row['list_items'], str) else row['list_items']
                data.append({
                    'type': row['type'],
                    'list_items': list_items,
                    'list_length': int(row['list_length']),
                    'answer': int(row['answer'])
                })
        
        logger.info(f"Loaded {len(data)} examples")
        return data
    
    def _get_model_evaluator(self, model_name: str, token: Optional[str] = None) -> ModelEvaluator:
        """Get appropriate evaluator for the model."""
        model_name_lower = model_name.lower()
        
        if "llama" in model_name_lower:
            return LlamaEvaluator(model_name, token)
        elif "phi" in model_name_lower:
            return PhiEvaluator(model_name, token)
        elif "qwen" in model_name_lower:
            return QwenEvaluator(model_name, token)
        else:
            # Default to generic pipeline approach
            return PhiEvaluator(model_name, token)
    
    def _create_prompt(self, example: Dict[str, Any]) -> str:
        """Create the counting prompt for the model."""
        return f"""Count how many words in this list match the type "{example['type']}".

List: {example['list_items']}

Respond with only the number in parentheses, like (0), (1), (2), etc."""
    
    def _create_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Create the message format for the model."""
        return [
            {
                "role": "system",
                "content": "You are a precise counting assistant. When given a list and a type, count how many items match that type. Always respond with ONLY the count in parentheses format: (0), (1), (2), etc. Never include explanations or other text."
            },
            {"role": "user", "content": prompt}
        ]
    
    def _extract_answer(self, response: str) -> int:
        """Extract numerical answer from model response."""
        text = response.strip()
        
        # Look for pattern (number)
        match = re.search(r'\((\d+)\)', text)
        if match:
            return int(match.group(1))
        
        # Look for standalone number
        match = re.search(r'^\s*(\d+)\s*$', text)
        if match:
            return int(match.group(1))
        
        # Look for any number in the text
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
        
        logger.warning(f"Could not extract answer from: '{text}'")
        return -1
    
    def evaluate_model(self, model_name: str, num_examples: Optional[int] = None,
                      token: Optional[str] = None) -> pd.DataFrame:
        """Evaluate a single model on the counting task."""
        
        # Get evaluator
        evaluator = self._get_model_evaluator(model_name, token)
        
        # Load model
        evaluator.load_model()
        
        # Prepare data
        num_examples = num_examples or len(self.dataset)
        num_examples = min(num_examples, len(self.dataset))
        
        logger.info(f"Evaluating {model_name} on {num_examples} examples...")
        
        results = []
        
        for i in tqdm(range(num_examples), desc=f"Evaluating {model_name}"):
            example = self.dataset[i]
            
            try:
                # Create prompt and generate response
                prompt = self._create_prompt(example)
                messages = self._create_messages(prompt)
                response = evaluator.generate_response(messages)
                
                # Extract answer
                extracted_answer = self._extract_answer(response)
                
                # Store result
                results.append({
                    'model': model_name,
                    'example_id': i,
                    'type': example['type'],
                    'list_items': str(example['list_items']),
                    'list_length': example['list_length'],
                    'expected_answer': example['answer'],
                    'model_response': response,
                    'extracted_answer': extracted_answer,
                    'correct': extracted_answer == example['answer']
                })
                
                # Progress logging
                if (i + 1) % 100 == 0:
                    accuracy = sum(r['correct'] for r in results) / len(results)
                    logger.info(f"Processed {i+1} examples. Current accuracy: {accuracy:.2%}")
            
            except Exception as e:
                logger.error(f"Error processing example {i}: {str(e)}")
                results.append({
                    'model': model_name,
                    'example_id': i,
                    'type': example['type'],
                    'list_items': str(example['list_items']),
                    'list_length': example['list_length'],
                    'expected_answer': example['answer'],
                    'model_response': f"ERROR: {str(e)}",
                    'extracted_answer': -1,
                    'correct': False
                })
        
        # Clean up model
        evaluator.cleanup()
        
        # Create DataFrame and save results
        df = pd.DataFrame(results)
        
        # Save individual model results
        model_filename = model_name.replace("/", "_").replace("-", "_")
        output_path = os.path.join(self.output_dir, f"{model_filename}_results.csv")
        df.to_csv(output_path, index=False)
        
        # Print summary
        accuracy = df['correct'].mean()
        logger.info(f"\n{model_name} Results:")
        logger.info(f"Overall accuracy: {accuracy:.2%}")
        logger.info(f"Correct: {df['correct'].sum()}/{len(df)}")
        logger.info(f"Results saved to: {output_path}")
        
        return df
    
    def evaluate_models(self, model_names: List[str], num_examples: Optional[int] = None,
                       tokens: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
        """Evaluate multiple models and create comparison."""
        
        if tokens is None:
            tokens = {}
        
        all_results = {}
        
        for model_name in model_names:
            try:
                token = tokens.get(model_name)
                df = self.evaluate_model(model_name, num_examples, token)
                all_results[model_name] = df
                
                # Memory cleanup between models
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
        
        # Create comparison summary
        self._create_comparison_summary(all_results)
        
        return all_results
    
    def _create_comparison_summary(self, all_results: Dict[str, pd.DataFrame]) -> None:
        """Create and save comparison summary."""
        
        comparison_data = []
        
        for model_name, df in all_results.items():
            # Overall statistics
            overall_acc = df['correct'].mean()
            total_examples = len(df)
            correct_count = df['correct'].sum()
            
            # Per-category statistics
            category_stats = df.groupby('type')['correct'].agg(['mean', 'count'])
            
            comparison_data.append({
                'model': model_name,
                'overall_accuracy': overall_acc,
                'total_examples': total_examples,
                'correct_count': correct_count,
                'category_stats': category_stats.to_dict()
            })
        
        # Save comparison
        comparison_df = pd.DataFrame([{
            'model': item['model'],
            'overall_accuracy': item['overall_accuracy'],
            'total_examples': item['total_examples'],
            'correct_count': item['correct_count']
        } for item in comparison_data])
        
        comparison_df = comparison_df.sort_values('overall_accuracy', ascending=False)
        
        comparison_path = os.path.join(self.output_dir, "benchmark_comparison.csv")
        comparison_df.to_csv(comparison_path, index=False)
        
        logger.info(f"\nModel Comparison Summary:")
        logger.info(f"\n{comparison_df}")
        logger.info(f"\nComparison saved to: {comparison_path}")


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark LLMs on counting task")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--models", nargs="+", required=True, help="Model names to evaluate")
    parser.add_argument("--num-examples", type=int, help="Number of examples to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--hf-token", type=str, help="HuggingFace token for gated models")
    
    args = parser.parse_args()
    
    # Setup tokens
    tokens = {}
    if args.hf_token:
        for model in args.models:
            if "llama" in model.lower() or "meta" in model.lower():
                tokens[model] = args.hf_token
    
    # Create benchmark and run evaluation
    benchmark = CountingBenchmark(args.dataset, args.output_dir)
    
    logger.info(f"Starting benchmark with models: {args.models}")
    results = benchmark.evaluate_models(args.models, args.num_examples, tokens)
    
    logger.info("Benchmark completed successfully!")
