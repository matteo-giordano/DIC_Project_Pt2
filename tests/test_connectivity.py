#!/usr/bin/env python3
"""
Test script for the ensure_connectivity function in the Dataset class.
This script generates different types of mazes, visualizes them before and after
ensuring connectivity, and shows the number of obstacles removed.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_generator import (
    PrimMazeGenerator,
    RecursiveDivisionMazeGenerator,
    WilsonMazeGenerator,
    TerrainGenerator,
    GaussianNoiseGenerator
)
from dataset import Dataset
from pathlib import Path

def test_connectivity():
    """
    Test the ensure_connectivity function on different maze types.
    """
    # Create output directory
    output_dir = Path("connectivity_test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Part 1: Verify that tree-based maze algorithms naturally create one component
    print("\n=== VERIFYING NATURALLY CONNECTED GENERATORS ===")
    naturally_connected_generators = [
        ("Prim", PrimMazeGenerator(31, 31)),
        ("Wilson", WilsonMazeGenerator(31, 31)),
        ("Recursive", RecursiveDivisionMazeGenerator(31, 31))
    ]
    
    for name, generator in naturally_connected_generators:
        # Generate a random maze
        maze = generator.generate_map()
        
        # Count connected components
        empty_cells, components = count_connected_components(maze)
        print(f"{name} maze has {components} connected component(s) with {empty_cells} empty cells")
        
        # Verify it has exactly one component
        if components != 1:
            print(f"  WARNING: {name} maze should have exactly 1 component, but has {components}!")
        else:
            print(f"  ✓ {name} maze correctly has 1 component as expected")
    
    # Part 2: Test the connectivity algorithm on mazes that might have multiple components
    print("\n=== TESTING CONNECTIVITY ALGORITHM ===")
    
    test_generators = [
        # Tree-based algorithms are naturally connected
        ("Prim", PrimMazeGenerator(31, 31)),
        ("Wilson", WilsonMazeGenerator(31, 31)),
        ("Recursive", RecursiveDivisionMazeGenerator(31, 31)),
        # These may have multiple components
        ("Terrain", TerrainGenerator(31, 31, blob_density=0.01)),
        ("Gaussian (0.65)", GaussianNoiseGenerator(31, 31, threshold=0.65)),
        ("Gaussian (0.75)", GaussianNoiseGenerator(31, 31, threshold=0.75))
    ]
    
    # Create a figure with original mazes on the left and connected mazes on the right
    fig, axes = plt.subplots(len(test_generators), 2, figsize=(12, 4 * len(test_generators)))
    
    for i, (name, generator) in enumerate(test_generators):
        print(f"\nTesting {name} maze generator")
        
        # Generate a random maze
        original_maze = generator.generate_map().copy()
        
        # For testing connectivity repair, use a separate copy
        test_maze = original_maze.copy()
        
        # Count connected components in the original maze
        empty_cells, components_original = count_connected_components(original_maze)
        print(f"Original maze has {components_original} connected component(s) with {empty_cells} empty cells")
        
        # Visualize the original maze
        ax1 = axes[i, 0]
        colored_maze = np.zeros((*original_maze.shape, 3))
        colored_maze[original_maze == 0] = (1, 1, 1)  # White for empty spaces
        colored_maze[original_maze == 1] = (0, 0, 0)  # Black for walls
        ax1.imshow(colored_maze)
        ax1.set_title(f"{name} - Original ({components_original} component{'s' if components_original > 1 else ''})")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Create dataset with the test maze
        dataset = Dataset("test")
        dataset.maps = [test_maze]
        dataset.metadata = [{"generator_type": name}]
        
        # Ensure connectivity if needed
        if components_original > 1:
            obstacles_removed = dataset.ensure_connectivity(0)
            print(f"Removed {obstacles_removed} obstacles")
            connected_maze = dataset.maps[0]
            
            # Count connected components after
            _, components_after = count_connected_components(connected_maze)
            print(f"After connectivity: {components_after} connected component(s)")
            
            # Visualize the connected maze
            ax2 = axes[i, 1]
            colored_maze = np.zeros((*connected_maze.shape, 3))
            colored_maze[connected_maze == 0] = (1, 1, 1)  # White for empty spaces
            colored_maze[connected_maze == 1] = (0, 0, 0)  # Black for walls
            
            # Highlight removed obstacles in red
            removed_mask = (original_maze == 1) & (connected_maze == 0)
            colored_maze[removed_mask] = (1, 0, 0)  # Red for removed obstacles
            
            ax2.imshow(colored_maze)
            ax2.set_title(f"{name} - Connected (removed: {obstacles_removed})")
            ax2.set_xticks([])
            ax2.set_yticks([])
        else:
            # No connectivity fix needed
            ax2 = axes[i, 1]
            ax2.imshow(colored_maze)  # Show the same maze
            ax2.set_title(f"{name} - Already connected")
            ax2.set_xticks([])
            ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / "connectivity_comparison.png")
    print(f"\nSaved visualization to {output_dir}/connectivity_comparison.png")
    
    # Part 3: Create a demonstration of artificially broken connectivity and its repair
    print("\n=== DEMONSTRATING CONNECTIVITY REPAIR ON ARTIFICIALLY BROKEN MAZES ===")
    
    # We'll only use the naturally connected generators for this test
    broken_generators = naturally_connected_generators.copy()
    
    # Create a figure
    fig2, axes2 = plt.subplots(len(broken_generators), 2, figsize=(12, 4 * len(broken_generators)))
    
    for i, (name, generator) in enumerate(broken_generators):
        print(f"\nBreaking connectivity of {name} maze")
        
        # Generate a random maze (should have 1 component)
        original_maze = generator.generate_map().copy()
        
        # Create a broken version
        broken_maze = original_maze.copy()
        
        # Add walls in a pattern that's likely to break connectivity
        height, width = broken_maze.shape
        print("Breaking connectivity for testing purposes...")
        for y in range(2, height-2, 4):
            for x in range(1, width-1, 2):
                broken_maze[y, x] = 1
                
        # Count connected components in the broken maze
        _, components_broken = count_connected_components(broken_maze)
        print(f"Broken maze has {components_broken} connected component(s)")
        
        # Visualize the broken maze
        ax1 = axes2[i, 0]
        colored_maze = np.zeros((*broken_maze.shape, 3))
        colored_maze[broken_maze == 0] = (1, 1, 1)  # White for empty spaces
        colored_maze[broken_maze == 1] = (0, 0, 0)  # Black for walls
        
        # Highlight added obstacles in blue
        added_mask = (original_maze == 0) & (broken_maze == 1)
        colored_maze[added_mask] = (0, 0, 1)  # Blue for added obstacles
        
        ax1.imshow(colored_maze)
        ax1.set_title(f"{name} - Broken ({components_broken} components)")
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Create dataset with the broken maze
        dataset = Dataset("test")
        dataset.maps = [broken_maze.copy()]
        dataset.metadata = [{"generator_type": name}]
        
        # Ensure connectivity
        obstacles_removed = dataset.ensure_connectivity(0)
        print(f"Removed {obstacles_removed} obstacles")
        repaired_maze = dataset.maps[0]
        
        # Count connected components after
        _, components_after = count_connected_components(repaired_maze)
        print(f"After repair: {components_after} connected component(s)")
        
        # Visualize the repaired maze
        ax2 = axes2[i, 1]
        colored_maze = np.zeros((*repaired_maze.shape, 3))
        colored_maze[repaired_maze == 0] = (1, 1, 1)  # White for empty spaces
        colored_maze[repaired_maze == 1] = (0, 0, 0)  # Black for walls
        
        # Highlight removed obstacles in red
        removed_mask = (broken_maze == 1) & (repaired_maze == 0)
        colored_maze[removed_mask] = (1, 0, 0)  # Red for removed obstacles
        
        ax2.imshow(colored_maze)
        ax2.set_title(f"{name} - Repaired (removed: {obstacles_removed})")
        ax2.set_xticks([])
        ax2.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_dir / "broken_maze_repair.png")
    print(f"Saved artificially broken maze visualization to {output_dir}/broken_maze_repair.png")
    
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
    test_connectivity() 