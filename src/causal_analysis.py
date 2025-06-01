"""
Causal Mediation Analysis for LLM Counting Tasks

This module implements counterfactual activation patching to investigate
whether specific neural network layers contain representations of running
counts during word list processing.

Example usage:
    analyzer = CausalMediationAnalyzer("microsoft/phi-4")
    results = analyzer.run_analysis("data/intervention_pairs.json")
"""

import torch
import json
import re
import gc
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InterventionDataGenerator:
    """Generate intervention pairs for causal mediation analysis."""
    
    def __init__(self, word_banks_path: str):
        """Initialize with word banks."""
        with open(word_banks_path, 'r') as f:
            self.word_banks = json.load(f)
        self.categories = list(self.word_banks.keys())
    
    def create_base_example(self, target_category: str, list_length: int = 7) -> Dict:
        """Create a base example for counting."""
        import random
        
        # Number of target words (1 to list_length-2)
        num_target_words = random.randint(1, max(2, list_length - 2))
        
        # Sample target words
        target_words = random.sample(
            self.word_banks[target_category],
            min(num_target_words, len(self.word_banks[target_category]))
        )
        
        # Sample distractor words from other categories
        distractor_words = []
        other_categories = [cat for cat in self.categories if cat != target_category]
        
        while len(distractor_words) < list_length - len(target_words):
            random_category = random.choice(other_categories)
            word = random.choice(self.word_banks[random_category])
            if word not in distractor_words and word not in target_words:
                distractor_words.append(word)
        
        # Create word list and shuffle
        word_list = target_words + distractor_words
        random.shuffle(word_list)
        
        # Record positions of target words
        target_positions = [i for i, word in enumerate(word_list) if word in target_words]
        
        return {
            'category': target_category,
            'word_list': word_list,
            'target_words': target_words,
            'distractor_words': distractor_words,
            'target_positions': target_positions,
            'count': len(target_words),
            'list_length': list_length
        }
    
    def create_intervention_pair(self, base_example: Dict) -> Optional[Dict]:
        """Create intervention by replacing a target word with a distractor."""
        import random
        
        target_positions = base_example['target_positions']
        if not target_positions:
            return None
        
        # Choose random target word position to intervene
        intervention_pos = random.choice(target_positions)
        original_word = base_example['word_list'][intervention_pos]
        
        # Choose replacement word from different category
        other_categories = [cat for cat in self.categories if cat != base_example['category']]
        replacement_category = random.choice(other_categories)
        
        # Get word not already in list
        available_words = [
            w for w in self.word_banks[replacement_category]
            if w not in base_example['word_list']
        ]
        
        if not available_words:
            return None
        
        intervention_word = random.choice(available_words)
        
        # Create intervention list
        intervention_list = base_example['word_list'].copy()
        intervention_list[intervention_pos] = intervention_word
        
        return {
            'category': base_example['category'],
            'original_list': base_example['word_list'],
            'intervention_list': intervention_list,
            'original_count': base_example['count'],
            'intervention_count': base_example['count'] - 1,  # One less target word
            'intervention_position': intervention_pos,
            'original_word': original_word,
            'intervention_word': intervention_word,
            'intervention_category': replacement_category
        }
    
    def generate_intervention_dataset(self, num_pairs: int = 1000, output_path: str = None) -> List[Dict]:
        """Generate dataset of intervention pairs."""
        import random
        
        intervention_pairs = []
        attempts = 0
        max_attempts = num_pairs * 3
        
        while len(intervention_pairs) < num_pairs and attempts < max_attempts:
            attempts += 1
            
            # Create base example
            category = random.choice(self.categories)
            list_length = random.randint(5, 8)
            base_example = self.create_base_example(category, list_length)
            
            # Create intervention pair
            intervention = self.create_intervention_pair(base_example)
            if intervention:
                intervention['pair_id'] = len(intervention_pairs)
                intervention_pairs.append(intervention)
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(intervention_pairs, f, indent=2)
            logger.info(f"Generated {len(intervention_pairs)} intervention pairs")
            logger.info(f"Saved to: {output_path}")
        
        return intervention_pairs


class CausalMediationAnalyzer:
    """Analyzer for causal mediation in counting tasks."""
    
    def __init__(self, model_name: str = "microsoft/phi-4", device: Optional[str] = None):
        """Initialize with model for analysis."""
        
        self.model_name = model_name
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        logger.info(f"Loading model {model_name}...")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        self.model.eval()
        
        # Get model architecture info
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            self.num_layers = len(self.model.model.layers)
            self.layer_attr = 'model.layers'
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            self.num_layers = len(self.model.transformer.h)
            self.layer_attr = 'transformer.h'
        else:
            raise ValueError("Cannot determine model architecture")
        
        logger.info(f"Model loaded with {self.num_layers} layers on {self.device}")
    
    def format_prompt(self, category: str, word_list: List[str]) -> str:
        """Format counting prompt with chat template."""
        
        problem = f"""Count how many words in this list match the type "{category}".

List: {word_list}

Respond with only the number in parentheses, like (0), (1), (2), etc."""
        
        messages = [
            {
                "role": "system", 
                "content": "You are a precise counting assistant. When given a list and a type, count how many items match that type. Always respond with ONLY the count in parentheses format: (0), (1), (2), etc. Never include explanations or other text."
            },
            {"role": "user", "content": problem}
        ]
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def extract_number_from_output(self, text: str) -> int:
        """Extract number from model output."""
        text = text.strip()
        
        # Look for pattern (number)
        match = re.search(r'\((\d+)\)', text)
        if match:
            return int(match.group(1))
        
        # Look for standalone number
        match = re.search(r'^\s*(\d+)\s*$', text)
        if match:
            return int(match.group(1))
        
        # Look for any number
        match = re.search(r'(\d+)', text)
        if match:
            return int(match.group(1))
        
        return -1
    
    def get_model_output(self, prompt: str) -> Tuple[int, str]:
        """Get model prediction and text output."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated text
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Extract number
        predicted_number = self.extract_number_from_output(generated_text)
        
        return predicted_number, generated_text
    
    def get_hidden_states(self, prompt: str) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        """Extract hidden states from all layers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract hidden states (skip embedding layer)
        hidden_states = {}
        all_hidden = outputs.hidden_states
        
        for layer_idx in range(self.num_layers):
            hidden_states[layer_idx] = all_hidden[layer_idx + 1].cpu()
        
        return hidden_states, inputs.input_ids
    
    def find_intervention_token_positions(self, prompt: str, word_list: List[str],
                                        intervention_position: int, target_word: str) -> List[int]:
        """Find token positions for intervention word."""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        full_tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Try to find target word tokens in the prompt
        target_word_tokens = self.tokenizer.tokenize(target_word)
        target_word_tokens_normalized = [token.lstrip('Ġ▁') for token in target_word_tokens]
        
        # Search for exact match
        for i in range(len(full_tokens) - len(target_word_tokens) + 1):
            slice_tokens = full_tokens[i:i+len(target_word_tokens)]
            normalized = [t.lstrip('Ġ▁') for t in slice_tokens]
            if normalized == target_word_tokens_normalized:
                return list(range(i, i + len(target_word_tokens)))
        
        # Fallback: fuzzy matching
        variants = [
            target_word.lower(),
            f"▁{target_word.lower()}",
            f"Ġ{target_word.lower()}"
        ]
        
        for i, token in enumerate(full_tokens):
            cleaned = token.lower().lstrip("Ġ▁")
            if cleaned in [v.lstrip("Ġ▁") for v in variants]:
                return [i]
        
        return []
    
    def patch_forward_pass(self, prompt: str, layer_idx: int,
                          patch_activation: torch.Tensor, 
                          patch_positions: List[int]) -> Tuple[int, str]:
        """Run forward pass with patched activations."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Get target layer
        if 'model.layers' in self.layer_attr:
            target_layer = self.model.model.layers[layer_idx]
        else:
            target_layer = self.model.transformer.h[layer_idx]
        
        # Store original forward method
        original_forward = target_layer.forward
        
        def patched_forward(hidden_states, *args, **kwargs):
            # Call original forward
            outputs = original_forward(hidden_states, *args, **kwargs)
            
            # Extract hidden states from output
            if isinstance(outputs, tuple):
                hidden_states_out = outputs[0]
                other_outputs = outputs[1:]
            else:
                hidden_states_out = outputs
                other_outputs = ()
            
            # Apply patch at specified positions
            for pos in patch_positions:
                if pos < hidden_states_out.shape[1] and pos < patch_activation.shape[1]:
                    patch_value = patch_activation[:, pos, :].to(hidden_states_out.device)
                    hidden_states_out[:, pos, :] = patch_value
            
            # Return in original format
            if isinstance(outputs, tuple):
                return (hidden_states_out,) + other_outputs
            else:
                return hidden_states_out
        
        # Apply patch
        target_layer.forward = patched_forward
        
        try:
            # Generate with patched activation
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Extract result
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            predicted_number = self.extract_number_from_output(generated_text)
            
        finally:
            # Restore original forward method
            target_layer.forward = original_forward
        
        return predicted_number, generated_text
    
    def calculate_effects(self, pair: Dict, layer_idx: int) -> Dict[str, Any]:
        """Calculate Total Effect (TE) and Indirect Effect (IE) for a pair and layer."""
        
        # Format prompts
        original_prompt = self.format_prompt(pair['category'], pair['original_list'])
        intervention_prompt = self.format_prompt(pair['category'], pair['intervention_list'])
        
        # Step 1: Get original and intervention outputs (for TE)
        original_num, original_text = self.get_model_output(original_prompt)
        intervention_num, intervention_text = self.get_model_output(intervention_prompt)
        
        # Total Effect
        TE = intervention_num - original_num
        
        # Step 2: Get hidden states
        original_hidden, _ = self.get_hidden_states(original_prompt)
        intervention_hidden, _ = self.get_hidden_states(intervention_prompt)
        
        # Step 3: Find positions to patch
        patch_positions = self.find_intervention_token_positions(
            intervention_prompt,
            pair['intervention_list'],
            pair['intervention_position'],
            pair['intervention_word']
        )
        
        # Step 4: Calculate Indirect Effect (IE)
        ie_original_positions = self.find_intervention_token_positions(
            original_prompt,
            pair['original_list'],
            pair['intervention_position'],
            pair['original_word']
        )
        
        ie_num, ie_text = self.patch_forward_pass(
            original_prompt,
            layer_idx,
            intervention_hidden[layer_idx],
            ie_original_positions
        )
        IE = ie_num - original_num
        
        # Collect results
        return {
            'pair_id': pair['pair_id'],
            'layer_idx': layer_idx,
            'TE': TE,
            'IE': IE,
            'original_output': original_num,
            'intervention_output': intervention_num,
            'ie_output': ie_num,
            'original_text': original_text.strip(),
            'intervention_text': intervention_text.strip(),
            'ie_text': ie_text.strip(),
            'expected_original': pair['original_count'],
            'expected_intervention': pair['intervention_count'],
            'original_correct': original_num == pair['original_count'],
            'intervention_correct': intervention_num == pair['intervention_count'],
            'patch_positions': patch_positions,
            'ie_positions': ie_original_positions,
            'num_patch_positions': len(patch_positions),
            'num_ie_positions': len(ie_original_positions),
            'category': pair['category'],
            'intervention_position': pair['intervention_position'],
            'original_word': pair['original_word'],
            'intervention_word': pair['intervention_word']
        }
    
    def calculate_effects_batch(self, pairs: List[Dict],
                               layers_to_test: Optional[List[int]] = None,
                               save_frequency: int = 10,
                               output_dir: str = "results") -> pd.DataFrame:
        """Calculate effects for multiple pairs and layers."""
        
        if layers_to_test is None:
            # Test middle and later layers by default
            layers_to_test = list(range(self.num_layers // 2, self.num_layers))
        
        results = []
        total_calculations = len(pairs) * len(layers_to_test)
        
        logger.info(f"Calculating effects for {len(pairs)} pairs and {len(layers_to_test)} layers")
        logger.info(f"Total calculations: {total_calculations}")
        
        with tqdm(total=total_calculations, desc="Calculating effects") as pbar:
            for pair_idx, pair in enumerate(pairs):
                for layer_idx in layers_to_test:
                    try:
                        # Calculate effects
                        effects = self.calculate_effects(pair, layer_idx)
                        results.append(effects)
                        
                    except Exception as e:
                        logger.error(f"Error with pair {pair['pair_id']}, layer {layer_idx}: {e}")
                        # Add failed result
                        results.append({
                            'pair_id': pair['pair_id'],
                            'layer_idx': layer_idx,
                            'error': str(e),
                            'TE': np.nan,
                            'IE': np.nan
                        })
                    
                    pbar.update(1)
                
                # Save intermediate results
                if (pair_idx + 1) % save_frequency == 0:
                    df_temp = pd.DataFrame(results)
                    temp_path = os.path.join(output_dir, f"causal_effects_intermediate_{pair_idx+1}.csv")
                    df_temp.to_csv(temp_path, index=False)
                    logger.info(f"Saved intermediate results ({pair_idx+1} pairs processed)")
                
                # Clear cache periodically
                if (pair_idx + 1) % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        # Create final dataframe
        results_df = pd.DataFrame(results)
        
        # Save final results
        final_path = os.path.join(output_dir, "causal_effects_results.csv")
        results_df.to_csv(final_path, index=False)
        logger.info(f"Saved final results to {final_path}")
        
        return results_df
    
    def run_analysis(self, pairs_path: str,
                    layers_to_test: Optional[List[int]] = None,
                    max_pairs: Optional[int] = None,
                    output_dir: str = "results") -> pd.DataFrame:
        """Complete pipeline to run effects calculation."""
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Load pairs
        logger.info(f"Loading pairs from {pairs_path}")
        with open(pairs_path, 'r') as f:
            pairs = json.load(f)
        
        # Limit pairs if specified
        if max_pairs:
            pairs = pairs[:max_pairs]
            logger.info(f"Using first {max_pairs} pairs")
        
        # Calculate effects
        results_df = self.calculate_effects_batch(pairs, layers_to_test, output_dir=output_dir)
        
        # Print summary statistics
        self._print_summary_stats(results_df)
        
        return results_df
    
    def _print_summary_stats(self, results_df: pd.DataFrame):
        """Print summary statistics of the results."""
        logger.info("\n=== Causal Mediation Analysis Summary ===")
        logger.info(f"Total calculations: {len(results_df)}")
        
        # Check for errors
        if 'error' in results_df.columns:
            error_count = results_df['error'].notna().sum()
            logger.info(f"Errors: {error_count}")
        
        # Basic statistics
        valid_results = results_df[results_df['TE'].notna()]
        logger.info(f"Valid results: {len(valid_results)}")
        
        if len(valid_results) > 0:
            logger.info("\nModel Accuracy:")
            orig_acc = valid_results['original_correct'].mean()
            int_acc = valid_results['intervention_correct'].mean()
            logger.info(f"  Original prompts: {orig_acc:.2%}")
            logger.info(f"  Intervention prompts: {int_acc:.2%}")
            
            logger.info("\nEffect Statistics:")
            logger.info(f"  Mean TE: {valid_results['TE'].mean():.3f}")
            logger.info(f"  Mean |TE|: {np.abs(valid_results['TE']).mean():.3f}")
            logger.info(f"  Mean IE: {valid_results['IE'].mean():.3f}")
            logger.info(f"  Mean |IE|: {np.abs(valid_results['IE']).mean():.3f}")
            
            # Effect by layer
            logger.info("\nTop 5 Layers by |IE|:")
            layer_effects = valid_results.groupby('layer_idx')[['TE', 'IE']].apply(
                lambda x: pd.Series({
                    'mean_TE': x['TE'].mean(),
                    'mean_abs_TE': np.abs(x['TE']).mean(),
                    'mean_IE': x['IE'].mean(),
                    'mean_abs_IE': np.abs(x['IE']).mean(),
                    'count': len(x)
                })
            ).reset_index()
            
            top_layers = layer_effects.nlargest(5, 'mean_abs_IE')
            logger.info(top_layers[['layer_idx', 'mean_abs_IE', 'mean_abs_TE', 'count']].to_string(index=False))


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run causal mediation analysis on counting task")
    parser.add_argument("--pairs-path", type=str, required=True, help="Path to intervention pairs JSON")
    parser.add_argument("--model", type=str, default="microsoft/phi-4", help="Model name")
    parser.add_argument("--max-pairs", type=int, help="Maximum number of pairs to process")
    parser.add_argument("--layers", nargs="+", type=int, help="Specific layers to test")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = CausalMediationAnalyzer(args.model)
    
    # Run analysis
    logger.info(f"Starting causal mediation analysis with {args.model}")
    results = analyzer.run_analysis(
        pairs_path=args.pairs_path,
        layers_to_test=args.layers,
        max_pairs=args.max_pairs,
        output_dir=args.output_dir
    )
    
    logger.info("Causal mediation analysis completed successfully!")
