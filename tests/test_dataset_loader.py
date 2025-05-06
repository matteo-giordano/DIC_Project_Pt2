#!/usr/bin/env python3
"""
Test script for loading datasets created by grid_generator.py
"""

from pathlib import Path
import argparse
from dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np

def find_most_recent_dataset(base_dir: str = None) -> str:
    """Find the most recently created dataset folder"""
    if base_dir is None:
        # First look in test_world folder
        base_dir = "test_world"
        if not Path(base_dir).exists():
            # Fall back to default_dataset folder
            base_dir = "default_dataset"
            if not Path(base_dir).exists():
                raise FileNotFoundError("No dataset folders found")
    
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Dataset base folder not found: {base_path}")
    
    # Find all dataset folders
    dataset_folders = list(base_path.glob("*"))
    if not dataset_folders:
        raise FileNotFoundError(f"No dataset folders found in {base_path}")
    
    # Sort by creation time (newest first)
    newest_folder = max(dataset_folders, key=lambda p: p.stat().st_mtime)
    return str(newest_folder)

def plot_sample_with_transforms(dataset):
    """
    Demonstrate sampling with random transformations
    """
    # Create a figure to show the original and transformed samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Get a single map without transformation (for reference)
    original_map, original_meta = dataset.sample(generator_type='prim')
    
    # Display the original in the first position
    ax = axes[0, 0]
    colored_map = np.zeros((*original_map.shape, 3))
    colored_map[original_map == 0] = [1, 1, 1]  # white
    colored_map[original_map == 1] = [0, 0, 0]  # black
    colored_map[original_map == 2] = [1, 0, 0]  # red
    
    ax.imshow(colored_map)
    ax.set_title("Original")
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Show 9 random transformations
    for i in range(1, 10):
        row = 0 if i < 5 else 1
        col = i % 5
        
        # Sample with random transformation
        transformed_map, transformed_meta = dataset.sample(
            generator_type='prim',
            random_transform=True
        )
        
        ax = axes[row, col]
        colored_map = np.zeros((*transformed_map.shape, 3))
        colored_map[transformed_map == 0] = [1, 1, 1]  # white
        colored_map[transformed_map == 1] = [0, 0, 0]  # black
        colored_map[transformed_map == 2] = [1, 0, 0]  # red
        
        ax.imshow(colored_map)
        transform = transformed_meta.get('transform', 'none')
        ax.set_title(f"Transform: {transform}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

def test_sampling(dataset):
    """Test the sampling functionality"""
    print("\n=== Sampling Test ===")
    
    # Get unique generator types
    generator_types = dataset.get_generator_types()
    print(f"Available generator types: {generator_types}")
    
    # Get unique dimensions
    dimensions = dataset.get_dimensions()
    print(f"Available dimensions: {dimensions}")
    
    # Sample a batch of maps for each generator type
    for gen_type in generator_types:
        print(f"\nSampling from {gen_type} generator...")
        
        # Sample a single map
        try:
            single_map, meta = dataset.sample(generator_type=gen_type)
            dims = f"{meta.get('width', '?')}x{meta.get('height', '?')}"
            print(f"Sampled {dims} map")
        except ValueError as e:
            print(f"Error: {e}")
            continue
        
        # Sample a batch
        try:
            batch, meta_list = dataset.sample_batch(batch_size=4, generator_type=gen_type)
            print(f"Sampled batch of {len(meta_list)} maps")
            if isinstance(batch, np.ndarray):
                print(f"Batch shape: {batch.shape}")
            else:
                print(f"Batch contains maps of different shapes")
        except ValueError as e:
            print(f"Error: {e}")
    
    # Demonstrate using filter_criteria
    print("\nDemonstrating filter_criteria parameter...")
    try:
        # Sample with more complex filtering
        filter_criteria = {'width': 21, 'height': 21}
        filtered_map, meta = dataset.sample(filter_criteria=filter_criteria)
        print(f"Sampled map with filter criteria {filter_criteria}")
        print(f"Map dimensions: {meta.get('width')}x{meta.get('height')}, generator: {meta.get('generator_type')}")
        
        # Sample batch with filter criteria
        batch, meta_list = dataset.sample_batch(batch_size=3, filter_criteria=filter_criteria)
        print(f"Sampled batch of {len(meta_list)} maps with filter criteria")
        for i, meta in enumerate(meta_list):
            print(f"  Map {i}: {meta.get('width')}x{meta.get('height')}, generator: {meta.get('generator_type')}")
    except ValueError as e:
        print(f"Error: {e}")
    
    # Demonstrate random transformations
    print("\nDemonstrating random transformations...")
    plot_sample_with_transforms(dataset)

def main():
    parser = argparse.ArgumentParser(description='Test dataset loading')
    parser.add_argument('--path', type=str, help='Path to dataset folder (default: auto-detect most recent)')
    parser.add_argument('--metadata-only', action='store_true', help='Load only metadata, not actual maps')
    parser.add_argument('--filter', type=str, help='Filter criteria as key=value pairs (e.g. "generator_type=prim width=21")')
    parser.add_argument('--visualize', action='store_true', help='Visualize sample maps')
    parser.add_argument('--count', type=int, default=4, help='Number of maps to visualize (default: 4)')
    parser.add_argument('--test-sampling', action='store_true', help='Test the sampling functionality')
    
    args = parser.parse_args()
    
    # Determine dataset path
    dataset_path = args.path
    if dataset_path is None:
        try:
            dataset_path = find_most_recent_dataset()
            print(f"Auto-detected most recent dataset: {dataset_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    
    # Load dataset
    dataset = Dataset()
    try:
        dataset.load(dataset_path, load_maps=not args.metadata_only)
        
        # Print summary
        dataset.summary()
        
        # Apply filter if specified
        if args.filter:
            criteria = {}
            for pair in args.filter.split():
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    # Try to convert value to int or float if possible
                    try:
                        value = int(value)
                    except ValueError:
                        try:
                            value = float(value)
                        except ValueError:
                            pass
                    criteria[key] = value
            
            if criteria:
                print(f"\nApplying filter: {criteria}")
                dataset = dataset.filter(criteria)
                dataset.summary()
        
        # Visualize if requested
        if args.visualize and not args.metadata_only:
            print("\nVisualizing sample maps...")
            dataset.visualize(max_maps=args.count)
        
        # Test sampling if requested
        if args.test_sampling and not args.metadata_only:
            test_sampling(dataset)
    
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main() 