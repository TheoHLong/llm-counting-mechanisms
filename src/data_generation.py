"""
LLM Counting Dataset Generator

This module generates synthetic datasets for evaluating Large Language Models
on word counting tasks across multiple semantic categories.

Example usage:
    generator = CountingDataGenerator()
    dataset = generator.create_dataset(size=5000)
    generator.save_dataset(dataset, "counting_dataset.csv")
"""

import json
import random
import csv
import os
from collections import Counter
from typing import Dict, List, Any, Optional


class CountingDataGenerator:
    """Generator for counting task datasets with multiple semantic categories."""
    
    def __init__(self):
        """Initialize with comprehensive word banks across 11 categories."""
        self.word_banks = {
            "fruit": [
                "apple", "banana", "cherry", "grape", "orange", "lemon", "peach", "pear", "plum", "berry",
                "melon", "kiwi", "mango", "lime", "date", "fig", "apricot", "coconut", "papaya", "guava",
                "pomegranate", "avocado", "strawberry", "blueberry", "raspberry", "blackberry", "cranberry",
                "watermelon", "cantaloupe", "honeydew", "grapefruit", "tangerine", "nectarine", "persimmon",
                "dragonfruit", "passionfruit", "starfruit", "lychee", "rambutan", "jackfruit", "durian",
                "elderberry", "gooseberry", "currant", "mulberry", "boysenberry", "lingonberry", "cloudberry"
            ],
            "animal": [
                "dog", "cat", "bird", "fish", "lion", "bear", "wolf", "deer", "mouse", "rabbit",
                "horse", "cow", "pig", "sheep", "goat", "chicken", "duck", "goose", "turkey", "eagle",
                "hawk", "owl", "parrot", "canary", "pigeon", "crow", "sparrow", "robin", "cardinal",
                "tiger", "leopard", "cheetah", "jaguar", "panther", "elephant", "rhino", "hippo", "giraffe",
                "zebra", "kangaroo", "koala", "panda", "monkey", "gorilla", "chimpanzee", "orangutan",
                "whale", "dolphin", "shark", "octopus", "squid", "crab", "lobster", "shrimp", "jellyfish",
                "turtle", "frog", "toad", "snake", "lizard", "crocodile", "alligator", "iguana"
            ],
            "vehicle": [
                "car", "bus", "truck", "bike", "plane", "boat", "train", "taxi", "van", "ship",
                "jet", "helicopter", "motorcycle", "scooter", "bicycle", "tricycle", "subway", "tram",
                "ferry", "yacht", "canoe", "kayak", "sailboat", "speedboat", "cruise", "cargo", "tanker",
                "ambulance", "firetruck", "police", "limousine", "convertible", "sedan", "hatchback",
                "wagon", "pickup", "trailer", "semi", "bulldozer", "excavator", "crane", "forklift",
                "tractor", "combine", "harvester", "snowplow", "garbage", "delivery", "rickshaw"
            ],
            "color": [
                "red", "blue", "green", "yellow", "black", "white", "pink", "brown", "gray", "purple",
                "orange", "silver", "gold", "violet", "cyan", "magenta", "turquoise", "indigo", "maroon",
                "navy", "teal", "lime", "olive", "aqua", "fuchsia", "coral", "salmon", "peach", "beige",
                "tan", "khaki", "ivory", "cream", "pearl", "platinum", "bronze", "copper", "rust",
                "crimson", "scarlet", "burgundy", "lavender", "lilac", "periwinkle", "azure", "cobalt"
            ],
            "body_part": [
                "head", "face", "eye", "nose", "mouth", "ear", "neck", "shoulder", "arm", "elbow",
                "wrist", "hand", "finger", "thumb", "nail", "chest", "back", "waist", "hip", "leg",
                "thigh", "knee", "shin", "ankle", "foot", "toe", "heel", "brain", "heart",
                "lung", "liver", "kidney", "stomach", "intestine", "muscle", "bone", "skin", "hair",
                "eyebrow", "eyelash", "cheek", "chin", "forehead", "temple", "jaw", "tooth", "tongue"
            ],
            "tool": [
                "hammer", "wrench", "screwdriver", "drill", "saw", "pliers", "knife", "scissors", "ruler",
                "tape", "level", "square", "chisel", "file", "sandpaper", "clamp", "vise", "anvil",
                "toolbox", "workbench", "ladder", "stepladder", "crowbar", "pickaxe", "shovel", "rake",
                "hoe", "spade", "trowel", "pruner", "shears", "mower", "trimmer", "blower", "chainsaw",
                "welder", "grinder", "router", "jigsaw", "bandsaw", "lathe", "press", "compressor"
            ],
            "clothing": [
                "shirt", "pants", "dress", "skirt", "jacket", "coat", "sweater", "hoodie", "blouse", "top",
                "jeans", "shorts", "trousers", "suit", "tie", "scarf", "hat", "cap", "gloves", "socks",
                "shoes", "boots", "sandals", "sneakers", "heels", "flats", "loafers", "slippers", "belt",
                "vest", "cardigan", "blazer", "tuxedo", "gown", "robe", "pajamas", "underwear", "bra",
                "bikini", "swimsuit", "uniform", "overalls", "jumpsuit", "romper", "tunic", "poncho"
            ],
            "sport": [
                "football", "basketball", "baseball", "soccer", "tennis", "golf", "hockey", "volleyball",
                "cricket", "rugby", "boxing", "wrestling", "swimming", "diving", "track", "marathon",
                "cycling", "skiing", "snowboard", "surfing", "skateboard", "bowling", "billiards", "darts",
                "archery", "fencing", "karate", "judo", "taekwondo", "gymnastics", "cheerleading", "dance",
                "yoga", "pilates", "aerobics", "crossfit", "weightlifting", "powerlifting", "bodybuilding",
                "climbing", "hiking", "camping", "fishing", "hunting", "sailing", "rowing", "kayaking"
            ],
            "building": [
                "house", "apartment", "mansion", "cottage", "cabin", "castle", "palace", "tower", "skyscraper",
                "office", "store", "shop", "mall", "market", "restaurant", "cafe", "bar", "hotel", "motel",
                "hospital", "clinic", "school", "university", "library", "museum", "theater", "cinema",
                "church", "temple", "mosque", "synagogue", "cathedral", "chapel", "monastery", "convent",
                "factory", "warehouse", "garage", "barn", "shed", "greenhouse", "lighthouse", "windmill",
                "bridge", "tunnel", "dam", "fort", "bunker", "observatory", "planetarium", "aquarium"
            ],
            "weather": [
                "sun", "rain", "snow", "wind", "cloud", "storm", "thunder", "lightning", "hail", "sleet",
                "fog", "mist", "drizzle", "shower", "downpour", "blizzard", "tornado", "hurricane", "cyclone",
                "typhoon", "drought", "flood", "frost", "ice", "dew", "humidity", "pressure", "temperature",
                "heat", "cold", "warm", "cool", "hot", "freezing", "mild", "severe", "gentle", "fierce",
                "calm", "breezy", "gusty", "windy", "stormy", "sunny", "cloudy", "overcast", "clear"
            ],
            "emotion": [
                "happy", "sad", "angry", "excited", "nervous", "calm", "peaceful", "anxious", "worried",
                "scared", "afraid", "brave", "confident", "shy", "proud", "ashamed", "guilty", "innocent",
                "curious", "bored", "interested", "fascinated", "amazed", "surprised", "shocked", "confused",
                "frustrated", "annoyed", "irritated", "pleased", "satisfied", "content", "grateful",
                "thankful", "hopeful", "optimistic", "pessimistic", "depressed", "elated", "ecstatic",
                "enthusiastic", "passionate", "loving", "caring", "compassionate", "empathetic", "sympathetic"
            ]
        }
    
    def generate_example(self, category: str, list_length: int = 7) -> Dict[str, Any]:
        """
        Generate a single counting example with uniform distribution.
        
        Args:
            category: Target word category to count
            list_length: Number of words in the list
            
        Returns:
            Dictionary containing the example data
        """
        target_words = self.word_banks[category]
        
        # Uniform distribution: equal probability for each possible count
        max_matches = min(len(target_words), list_length)
        num_matches = random.randint(0, max_matches)
        
        # Select target category words
        matches = random.sample(target_words, num_matches) if num_matches > 0 else []
        
        # Fill remaining slots with words from other categories
        remaining_slots = list_length - len(matches)
        fillers = []
        other_categories = [cat for cat in self.word_banks if cat != category]
        
        for _ in range(remaining_slots):
            other_cat = random.choice(other_categories)
            other_word = random.choice(self.word_banks[other_cat])
            fillers.append(other_word)
        
        # Combine and shuffle
        word_list = matches + fillers
        random.shuffle(word_list)
        
        return {
            'type': category,
            'list_items': word_list,
            'list_length': len(word_list),
            'answer': len(matches)
        }
    
    def create_dataset(self, size: int = 5000) -> List[Dict[str, Any]]:
        """
        Generate complete dataset with specified number of examples.
        
        Args:
            size: Number of examples to generate
            
        Returns:
            List of example dictionaries
        """
        examples = []
        categories_list = list(self.word_banks.keys())
        list_lengths = [5, 6, 7, 8, 9, 10]  # Variable list lengths
        
        for i in range(size):
            category = random.choice(categories_list)
            list_length = random.choice(list_lengths)
            
            try:
                example = self.generate_example(category, list_length)
                examples.append(example)
            except Exception as e:
                print(f"Error generating example {i}: {e}")
                continue
        
        return examples
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate dataset quality and return statistics.
        
        Args:
            dataset: List of generated examples
            
        Returns:
            Dictionary of validation statistics
        """
        stats = {
            'total_examples': len(dataset),
            'answer_distribution': Counter([ex['answer'] for ex in dataset]),
            'category_distribution': Counter([ex['type'] for ex in dataset]),
            'length_distribution': Counter([ex['list_length'] for ex in dataset])
        }
        
        # Check for duplicates
        examples_key = [(ex['type'], str(ex['list_items'])) for ex in dataset]
        stats['duplicates'] = len(examples_key) - len(set(examples_key))
        
        return stats
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save dataset to CSV file.
        
        Args:
            dataset: List of examples to save
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        fieldnames = ['type', 'list_items', 'list_length', 'answer']
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for example in dataset:
                # Convert list to string format for CSV
                row = example.copy()
                row['list_items'] = ', '.join([f"'{word}'" for word in example['list_items']])
                writer.writerow(row)
        
        print(f"Dataset saved to: {filepath}")
    
    def save_word_banks(self, filepath: str) -> None:
        """
        Save word banks to JSON file for reproducibility.
        
        Args:
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.word_banks, f, indent=2)
        
        print(f"Word banks saved to: {filepath}")


# Example usage and command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate counting task dataset")
    parser.add_argument("--size", type=int, default=5000, help="Number of examples to generate")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create generator and dataset
    generator = CountingDataGenerator()
    
    print(f"Generating dataset with {args.size} examples...")
    dataset = generator.create_dataset(args.size)
    
    # Validate dataset
    stats = generator.validate_dataset(dataset)
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total examples: {stats['total_examples']}")
    print(f"Duplicates: {stats['duplicates']}")
    print(f"Categories: {len(stats['category_distribution'])}")
    print(f"Answer range: {min(stats['answer_distribution'])} to {max(stats['answer_distribution'])}")
    
    # Save files
    output_dir = args.output_dir
    dataset_path = os.path.join(output_dir, f"counting_dataset_{args.size}.csv")
    word_banks_path = os.path.join(output_dir, "word_banks.json")
    
    generator.save_dataset(dataset, dataset_path)
    generator.save_word_banks(word_banks_path)
    
    # Show sample examples
    print(f"\n=== Sample Examples ===")
    for i in range(min(3, len(dataset))):
        ex = dataset[i]
        print(f"\nExample {i+1}:")
        print(f"  Type: {ex['type']}")
        print(f"  List: {ex['list_items']}")
        print(f"  Answer: {ex['answer']}")
