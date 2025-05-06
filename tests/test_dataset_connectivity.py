#!/usr/bin/env python3
"""
Test script to load maps from an existing dataset, check for connectivity issues,
apply the ensure_connectivity algorithm, and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataset import Dataset
from pathlib import Path
import random

def test_dataset_connectivity(dataset_path, max_maps=16):
    """
    Test the ensure_connectivity function on maps from an existing dataset.
    
    Args:
        dataset_path (str): Path to the dataset
        max_maps (int): Maximum number of maps to process
    """
    print(f"Loading dataset from: {dataset_path}")
    
    # Create output directory
    output_dir = Path("connectivity_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Load the dataset
    dataset = Dataset()
    dataset.load(dataset_path)
    
    print(f"Loaded dataset with {len(dataset.maps)} maps")
    
    # Find maps with multiple components
    disconnected_maps = []
    
    for i, maze in enumerate(dataset.maps):
        if maze is not None:
            # Skip maps that are too large for quick testing
            if max(maze.shape) > 100:
                continue
                
            # Count components
            empty_cells, components = count_connected_components(maze)
            
            if components > 1:
                disconnected_maps.append((i, components, empty_cells))
                if len(disconnected_maps) >= max_maps:
                    break
    
    print(f"Found {len(disconnected_maps)} maps with multiple components")
    
    if not disconnected_maps:
        # If we couldn't find any disconnected maps, just sample from the dataset
        indices = random.sample(range(len(dataset.maps)), min(max_maps, len(dataset.maps)))
        sample_dataset = Dataset()
        
        for idx in indices:
            if dataset.maps[idx] is not None:
                sample_dataset.maps.append(dataset.maps[idx].copy())
                sample_dataset.metadata.append(dataset.metadata[idx].copy())
        
        # Artificially break connectivity in these maps for demonstration
        for i in range(len(sample_dataset.maps)):
            maze = sample_dataset.maps[i]
            height, width = maze.shape
            
            # Add walls in a pattern that's likely to break connectivity
            for y in range(2, height-2, 4):
                for x in range(1, width-1, 2):
                    if maze[y, x] == 0:  # Only if it's currently a path
                        maze[y, x] = 1
            
            # Count components after breaking
            _, components = count_connected_components(maze)
            disconnected_maps.append((i, components, 0))
        
        dataset = sample_dataset
    
    # Create a figure for visualization
    num_maps = len(disconnected_maps)
    fig, axes = plt.subplots(num_maps, 2, figsize=(12, 4 * num_maps))
    
    # Make sure axes is always 2D even for a single map
    if num_maps == 1:
        axes = axes.reshape(1, 2)
    
    print("\nProcessing disconnected maps...")
    
    for i, (idx, components, empty_cells) in enumerate(disconnected_maps):
        print(f"\nMap {idx}: {components} components with {empty_cells} empty cells")
        
        # Get the original maze
        original_maze = dataset.maps[idx].copy()
        
        # Make a copy for connectivity repair
        maze_copy = original_maze.copy()
        
        # Create a temporary dataset for this maze
        temp_dataset = Dataset()
        temp_dataset.maps = [maze_copy]
        if idx < len(dataset.metadata):
            temp_dataset.metadata = [dataset.metadata[idx]]
        else:
            temp_dataset.metadata = [{"generator_type": "unknown"}]
        
        # Visualize the original maze
        ax1 = axes[i, 0]
        colored_maze = np.zeros((*original_maze.shape, 3))
        colored_maze[original_maze == 0] = (1, 1, 1)  # White for empty spaces
        colored_maze[original_maze == 1] = (0, 0, 0)  # Black for walls
        ax1.imshow(colored_maze)
        
        meta = temp_dataset.metadata[0]
        generator_type = meta.get('generator_type', 'unknown')
        width = meta.get('width', original_maze.shape[1])
        height = meta.get('height', original_maze.shape[0])
        
        ax1.set_title(f"{generator_type} {width}x{height} - Original ({components} components)")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Ensure connectivity
        obstacles_removed = temp_dataset.ensure_connectivity(0)
        print(f"Removed {obstacles_removed} obstacles")
        fixed_maze = temp_dataset.maps[0]
        
        # Count components after fixing
        _, components_after = count_connected_components(fixed_maze)
        print(f"After fixing: {components_after} component(s)")
        
        # Visualize the fixed maze
        ax2 = axes[i, 1]
        colored_maze = np.zeros((*fixed_maze.shape, 3))
        colored_maze[fixed_maze == 0] = (1, 1, 1)  # White for empty spaces
        colored_maze[fixed_maze == 1] = (0, 0, 0)  # Black for walls
        
        # Highlight removed obstacles in red
        removed_mask = (original_maze == 1) & (fixed_maze == 0)
        colored_maze[removed_mask] = (1, 0, 0)  # Red for removed obstacles
        
        ax2.imshow(colored_maze)
        ax2.set_title(f"Fixed (removed: {obstacles_removed}, components: {components_after})")
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_connectivity_test.png")
    print(f"\nSaved visualization to {output_dir}/dataset_connectivity_test.png")
    
    plt.show()

def count_connected_components(maze):
    """
    Count the number of connected components in a maze.
    
    Args:
        maze (np.ndarray): The maze to analyze
        
    Returns:
        tuple: (number of empty cells, number of connected components)
    """
    from collections import deque
    
    height, width = maze.shape
    visited = set()
    components = 0
    empty_cells = 0
    
    for y in range(height):
        for x in range(width):
            if maze[y, x] == 0 and (y, x) not in visited:
                # Found a new component
                components += 1
                queue = deque([(y, x)])
                visited.add((y, x))
                component_size = 1  # Start with 1 for the current cell
                
                # BFS to find all cells in this component
                while queue:
                    cy, cx = queue.popleft()
                    
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        
                        if (0 <= ny < height and 0 <= nx < width and 
                            maze[ny, nx] == 0 and (ny, nx) not in visited):
                            queue.append((ny, nx))
                            visited.add((ny, nx))
                            component_size += 1
                
                empty_cells += component_size
    
    return empty_cells, components

if __name__ == "__main__":
    dataset_path = "/home/barry/Desktop/uni/dataChallenge/dataChallengeG15/datachallengeg15/world/test_world/test_world_20250504_190646"
    test_dataset_connectivity(dataset_path) 