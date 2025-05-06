#!/usr/bin/env python3
"""
Test script for the Gaussian Noise Maze Generator.
This script generates and displays a set of mazes using the GaussianNoiseGenerator
with different threshold and sigma values.
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_generator import GaussianNoiseGenerator, plot_mazes
from pathlib import Path

def test_gaussian_noise_generator():
    """
    Generate and display a set of Gaussian noise mazes with different parameters.
    """
    # Define a grid of parameter combinations to test
    thresholds = [0.55, 0.6, 0.65, 0.7, 0.75]
    sigmas = [(0.5, 0.5), (1.0, 1.0), (1.5, 1.5), (2.0, 2.0)]
    
    # Size of the mazes
    width, height = 31, 31
    
    # Create a list to hold all generated mazes
    all_mazes = []
    labels = []
    
    # Generate mazes with different parameter combinations
    for threshold in thresholds:
        for sigma_x, sigma_y in sigmas:
            # Create a generator with the current parameters
            generator = GaussianNoiseGenerator(
                width=width, 
                height=height, 
                threshold=threshold,
                sigma_x=sigma_x,
                sigma_y=sigma_y
            )
            
            # Generate a maze
            maze = generator.generate_map()
            all_mazes.append(maze)
            
            # Create a label for this maze
            label = f"t={threshold}, σ=({sigma_x},{sigma_y})"
            labels.append(label)
    
    # Plot all the mazes in a grid
    fig, axes = plt.subplots(len(thresholds), len(sigmas), figsize=(15, 15))
    
    for idx, (maze, label) in enumerate(zip(all_mazes, labels)):
        row = idx // len(sigmas)
        col = idx % len(sigmas)
        
        ax = axes[row, col]
        
        # Create a colored grid
        colored_maze = np.zeros((*maze.shape, 3))
        colored_maze[maze == 0] = plt.matplotlib.colors.to_rgb('white')  # Path
        colored_maze[maze == 1] = plt.matplotlib.colors.to_rgb('black')  # Wall
        
        ax.imshow(colored_maze)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = Path("gaussian_test_output")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "gaussian_maze_samples.png")
    
    print(f"Generated {len(all_mazes)} Gaussian noise mazes")
    print(f"Saved visualization to {output_dir}/gaussian_maze_samples.png")
    plt.show()

if __name__ == "__main__":
    test_gaussian_noise_generator() 