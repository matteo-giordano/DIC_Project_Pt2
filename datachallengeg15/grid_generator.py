import numpy as np
from typing import List, Tuple, Union, Dict, Any
import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import yaml
import json
import itertools
from tqdm import tqdm
from datetime import datetime
from pathlib import Path
from scipy.ndimage import gaussian_filter


class DatasetGenerator:
    def __init__(self, config_file: str = 'dataset_config.yaml'):
        """
        Initialize the dataset generator with a configuration file.
        
        Args:
            config_file (str): Path to the YAML configuration file
        """
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.dataset = []
        self.metadata = []  # Store metadata for each generated map
    
    def _load_config(self) -> Dict:
        """
        Load configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        try:
            # Print the actual file path we're trying to open
            print(f"Trying to open config file: {self.config_file.absolute()}")
            
            if not self.config_file.exists():
                print(f"File does not exist: {self.config_file.absolute()}")
                
            with self.config_file.open('r') as file:
                config = yaml.safe_load(file)
                
                # Verify loaded config is not None
                if config is None:
                    print(f"Warning: Config file {self.config_file} was loaded but is empty or invalid YAML")
                    return {
                        'name': 'default_dataset',
                        'ensure_connected': True,
                        'generators': {
                            'prim': {
                                'count_per_config': 2,
                                'width': [21],
                                'height': [21]
                            }
                        }
                    }
                
                # Debug: print what was loaded
                print(f"Loaded config: {config}")
                
                return config
                
        except FileNotFoundError:
            print(f"Config file {self.config_file} not found. Using default configuration.")
            return {
                'name': 'default_dataset',
                'ensure_connected': True,
                'generators': {
                    'prim': {
                        'count_per_config': 2,
                        'width': [21],
                        'height': [21]
                    }
                }
            }
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return {
                'name': 'default_dataset',
                'ensure_connected': True,
                'generators': {
                    'prim': {
                        'count_per_config': 2,
                        'width': [21],
                        'height': [21]
                    }
                }
            }

    def _expand_config(self) -> List[Dict]:
        """
        Expand the configuration to create all possible parameter combinations.
        
        Returns:
            List[Dict]: List of expanded configuration dictionaries
        """
        expanded_configs = []
        
        for generator_type, params in self.config.get('generators', {}).items():
            count_per_config = params.get('count_per_config', 1)
            
            # Get lists of parameters or convert single values to lists
            param_lists = {}
            for key, value in params.items():
                if key != 'count_per_config':
                    param_lists[key] = value if isinstance(value, list) else [value]
            
            # Get all parameter combinations
            param_names = list(param_lists.keys())
            param_values = list(param_lists.values())
            
            for combination in itertools.product(*param_values):
                config_dict = {
                    'generator_type': generator_type,
                    'count': count_per_config
                }
                
                for i, name in enumerate(param_names):
                    config_dict[name] = combination[i]
                
                expanded_configs.append(config_dict)
        
        return expanded_configs
    
    def generate_dataset(self) -> List[np.ndarray]:
        """
        Generate a dataset of maps based on the configuration.
        
        Returns:
            List[np.ndarray]: List of generated maps
        """
        self.dataset = []
        self.metadata = []
        expanded_configs = self._expand_config()
        
        print(f"Generating dataset with {len(expanded_configs)} different configurations...")
        
        for config in tqdm(expanded_configs, desc="Generating maps"):
            generator_type = config.pop('generator_type')
            count = config.pop('count')
            width = config.pop('width')
            height = config.pop('height')
            
            # Create the appropriate generator
            if generator_type.lower() == 'prim':
                generator = PrimMazeGenerator(width, height, **config)
            elif generator_type.lower() == 'recursive':
                generator = RecursiveDivisionMazeGenerator(width, height, **config)
            elif generator_type.lower() == 'wilson':
                generator = WilsonMazeGenerator(width, height, **config)
            elif generator_type.lower() == 'terrain':
                generator = TerrainGenerator(width, height, **config)
            elif generator_type.lower() == 'gaussian':
                generator = GaussianNoiseGenerator(width, height, **config)
            else:
                print(f"Unknown generator type: {generator_type}, skipping...")
                continue
            
            # Generate maps
            for _ in range(count):
                map_data = generator.generate_map()
                self.dataset.append(map_data)
                
                # Store metadata for this map
                meta = {
                    'generator_type': generator_type,
                    'width': width,
                    'height': height,
                    'shape': map_data.shape,
                    'generation_time': datetime.now().isoformat(),
                    **config  # Include any additional parameters
                }
                self.metadata.append(meta)
        
        print(f"Generated {len(self.dataset)} maps in total")
        
        # Ensure connectivity if specified in the config
        ensure_connected = self.config.get('ensure_connected', False)
        if ensure_connected:
            print("Ensuring connectivity for all mazes...")
            from dataset import Dataset
            temp_dataset = Dataset()
            temp_dataset.maps = self.dataset
            temp_dataset.metadata = self.metadata
            
            obstacles_removed = temp_dataset.ensure_connectivity()
            print(f"Removed {obstacles_removed} obstacles to ensure connectivity")
            
            # Update our dataset with the connected mazes
            self.dataset = temp_dataset.maps
            self.metadata = temp_dataset.metadata
            
            # Update metadata to indicate connectivity was ensured
            for meta in self.metadata:
                meta['connectivity_ensured'] = True
                
        return self.dataset
    
    def deduplicate_dataset(self) -> int:
        """
        Remove duplicate mazes from the dataset.
        
        A maze is considered a duplicate if it is identical to another maze
        after accounting for rotations and flips.
        
        Returns:
            int: Number of duplicates removed
        """
        if not self.dataset:
            print("No dataset to deduplicate")
            return 0
        
        initial_count = len(self.dataset)
        print(f"Deduplicating dataset with {initial_count} mazes...")
        
        # Create a list of tuples (map, metadata, canonical_form)
        dataset_items = []
        for i, maze in enumerate(self.dataset):
            # Convert the maze to a canonical form for comparison
            # We'll use the lexicographically smallest representation
            # after all possible rotations and flips
            canonical_form = self._get_canonical_form(maze)
            dataset_items.append((maze, self.metadata[i], canonical_form))
        
        # Sort by canonical form to group duplicates together
        dataset_items.sort(key=lambda x: x[2])
        
        # Keep only unique mazes
        unique_items = []
        prev_canonical = None
        
        for maze, metadata, canonical in dataset_items:
            if canonical != prev_canonical:
                unique_items.append((maze, metadata))
                prev_canonical = canonical
            else:
                # This is a duplicate - skip it
                pass
        
        # Update the dataset and metadata with unique items
        self.dataset = [item[0] for item in unique_items]
        self.metadata = [item[1] for item in unique_items]
        
        # Update metadata to indicate deduplication
        for meta in self.metadata:
            meta['deduplicated'] = True
        
        removed_count = initial_count - len(self.dataset)
        print(f"Removed {removed_count} duplicate mazes, {len(self.dataset)} unique mazes remain")
        
        return removed_count
    
    def _get_canonical_form(self, maze: np.ndarray) -> tuple:
        """
        Convert a maze to its canonical form for comparison.
        Returns the lexicographically smallest representation after all
        possible rotations and flips.
        
        Args:
            maze (np.ndarray): The maze to convert
            
        Returns:
            tuple: A tuple representation of the canonical form
        """
        # Generate all possible orientations (rotations and flips)
        variants = []
        
        # Get all rotations and flips
        for k in range(4):  # 0, 90, 180, 270 degrees
            rotated = np.rot90(maze, k)
            # Convert to tuple for comparisons (immutable)
            variants.append(tuple(map(tuple, rotated)))
            
            # Also add flipped versions
            flipped = np.fliplr(rotated)
            variants.append(tuple(map(tuple, flipped)))
        
        # Return the lexicographically smallest variant
        return min(variants)
    
    def augment_dataset(self) -> List[np.ndarray]:
        """
        Augment the dataset by generating rotated and flipped versions of each map.
        
        Returns:
            List[np.ndarray]: Augmented dataset
        """
        if not self.dataset:
            self.generate_dataset()
        
        augmented_dataset = []
        augmented_metadata = []
        print("Augmenting dataset...")
        
        for idx, map_data in enumerate(tqdm(self.dataset, desc="Augmenting maps")):
            # Use the base class augment_map method
            augmented_maps = MazeGenerator.augment_map(map_data)
            
            # Add the augmented maps to our dataset
            augmented_dataset.extend(augmented_maps)
            
            # Duplicate and update the metadata for each augmented map
            base_meta = self.metadata[idx].copy()
            for i, _ in enumerate(augmented_maps):
                aug_meta = base_meta.copy()
                if i % 2 == 0:
                    aug_meta['augmentation'] = f'rotation_{i//2 * 90}deg'
                else:
                    aug_meta['augmentation'] = f'rotation_{i//2 * 90}deg_flipped'
                aug_meta['original_index'] = idx
                augmented_metadata.append(aug_meta)
        
        self.dataset = augmented_dataset
        self.metadata = augmented_metadata
        
        print(f"Augmented dataset contains {len(augmented_dataset)} maps")
        return augmented_dataset
    
    def save_dataset(self, base_folder: str = None):
        """
        Save the dataset to a folder with metadata.
        
        Args:
            base_folder (str, optional): Base folder to save the dataset to.
                If None, uses the name from the config file.
        """
        if not self.dataset:
            self.generate_dataset()
        
        # Get dataset name from config or use default
        dataset_name = self.config.get('name', 'default_dataset')
        
        # Create the folder structure
        if base_folder is None:
            output_dir = Path(dataset_name)
        else:
            output_dir = Path(base_folder)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_folder = output_dir / f"{dataset_name}_{timestamp}"
        maps_folder = dataset_folder / "maps"
        
        # Create folders if they don't exist
        maps_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving dataset to {dataset_folder}...")
        
        # Save each map with its index
        for i, map_data in enumerate(tqdm(self.dataset, desc="Saving maps")):
            map_path = maps_folder / f"map_{i:05d}.npy"
            np.save(map_path, map_data)
            
            # Add file path to metadata
            rel_path = map_path.relative_to(dataset_folder)
            self.metadata[i]['file_path'] = str(rel_path)
        
        # Check if all maps have the same shape before combining
        shapes = [map_data.shape for map_data in self.dataset]
        if len(set(shapes)) == 1:  # All maps have the same shape
            # Save combined dataset for convenience
            combined_path = dataset_folder / "combined_dataset.npy"
            np.save(combined_path, np.array(self.dataset))
            print(f"- Combined dataset saved to {combined_path}")
        else:
            print("- Maps have different shapes, skipping combined dataset")
            combined_path = "N/A (maps have different shapes)"
        
        # Save metadata as JSON
        metadata_path = dataset_folder / "metadata.json"
        with metadata_path.open('w') as f:
            json.dump({
                'dataset_name': dataset_name,
                'creation_time': timestamp,
                'config': self.config,
                'maps': self.metadata
            }, f, indent=2)
        
        # Save a sample visualization
        sample_size = min(16, len(self.dataset))
        sample_indices = random.sample(range(len(self.dataset)), sample_size)
        samples = [self.dataset[i] for i in sample_indices]
        
        fig = plot_mazes(samples, figsize=(15, 15))
        fig.savefig(dataset_folder / "sample_visualization.png")
        plt.close(fig)
        
        print(f"Dataset saved to {dataset_folder}")
        print(f"- {len(self.dataset)} maps saved")
        print(f"- Metadata saved to {metadata_path}")
        print(f"- Sample visualization saved to sample_visualization.png")


class MazeGenerator(ABC):
    def __init__(self, width: int, height: int):
        """
        Initialize the maze generator with given dimensions.
        
        Args:
            width (int): Width of the maze (including borders)
            height (int): Height of the maze (including borders)
        """
        self.width = width
        self.height = height
        self.map = np.ones((height, width), dtype=np.int8)  # Initialize with walls (1s)
    
    @abstractmethod
    def generate_map(self) -> np.ndarray:
        """
        Generate a random maze using the specific algorithm.
        
        Returns:
            np.ndarray: Generated maze with borders (1s) and empty spaces (0s)
        """
        pass

    @staticmethod
    def augment_map(map_data: np.ndarray) -> List[np.ndarray]:
        """
        Generate all unique rotated and mirrored versions of the map.
        Returns 8 versions: original, 3 rotations, and each with a horizontal flip.
        
        Args:
            map_data (np.ndarray): The map to augment
            
        Returns:
            List[np.ndarray]: List of augmented maps
        """
        augmentations = []
        for k in range(4):  # 0, 90, 180, 270 rotations
            rotated = np.rot90(map_data, k)
            augmentations.append(rotated)
            augmentations.append(np.fliplr(rotated))
        return augmentations


class PrimMazeGenerator(MazeGenerator):
    def _get_neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells for a given cell.
        
        Args:
            cell (Tuple[int, int]): Current cell coordinates
            
        Returns:
            List[Tuple[int, int]]: List of valid neighboring cells
        """
        x, y = cell
        neighbors = []
        
        # Check all four directions
        for dx, dy in [(0, 2), (2, 0), (0, -2), (-2, 0)]:
            nx, ny = x + dx, y + dy
            if 0 < nx < self.width - 1 and 0 < ny < self.height - 1:
                neighbors.append((nx, ny))
                
        return neighbors
    
    def _get_wall_between(self, cell1: Tuple[int, int], cell2: Tuple[int, int]) -> Tuple[int, int]:
        """
        Get the wall cell between two cells.
        
        Args:
            cell1 (Tuple[int, int]): First cell coordinates
            cell2 (Tuple[int, int]): Second cell coordinates
            
        Returns:
            Tuple[int, int]: Wall cell coordinates
        """
        return ((cell1[0] + cell2[0]) // 2, (cell1[1] + cell2[1]) // 2)
    
    def generate_map(self) -> np.ndarray:
        """
        Generate a random maze using Prim's algorithm.
        
        Returns:
            np.ndarray: Generated maze with borders (1s) and empty spaces (0s)
        """
        # Start with a grid of walls
        self.map = np.ones((self.height, self.width), dtype=np.int8)
        
        # Start from a random cell (must be odd coordinates)
        start_x = random.randrange(1, self.width - 1, 2)
        start_y = random.randrange(1, self.height - 1, 2)
        start_cell = (start_x, start_y)
        
        # Initialize the maze with the start cell
        self.map[start_y, start_x] = 0
        
        # List of walls to process
        walls = []
        for neighbor in self._get_neighbors(start_cell):
            walls.append((start_cell, neighbor))
        
        while walls:
            # Randomly select a wall
            cell1, cell2 = random.choice(walls)
            walls.remove((cell1, cell2))
            
            # If only one of the cells is visited
            if self.map[cell1[1], cell1[0]] != self.map[cell2[1], cell2[0]]:
                # Make the unvisited cell a path
                if self.map[cell1[1], cell1[0]] == 1:
                    self.map[cell1[1], cell1[0]] = 0
                    current_cell = cell1
                else:
                    self.map[cell2[1], cell2[0]] = 0
                    current_cell = cell2
                
                # Remove the wall between the cells
                wall = self._get_wall_between(cell1, cell2)
                self.map[wall[1], wall[0]] = 0
                
                # Add new walls to the list
                for neighbor in self._get_neighbors(current_cell):
                    if self.map[neighbor[1], neighbor[0]] == 1:
                        walls.append((current_cell, neighbor))
        
        return self.map


class RecursiveDivisionMazeGenerator(MazeGenerator):
    def _get_wall_position(self, low: int, high: int) -> int:
        return random.randrange(low + 1, high, 2)

    def _get_passage_position(self, low: int, high: int) -> int:
        return random.randrange(low, high + 1, 2)

    def _divide(self, x1: int, x2: int, y1: int, y2: int):
        if x2 - x1 < 2 or y2 - y1 < 2:
            return

        # Choose vertical and horizontal walls
        wx = self._get_wall_position(x1, x2)
        wy = self._get_wall_position(y1, y2)

        # Draw walls
        for x in range(x1, x2 + 1):
            self.map[wy, x] = 1
        for y in range(y1, y2 + 1):
            self.map[y, wx] = 1

        # Choose 3 of 4 wall segments to have passages
        directions = ['top', 'bottom', 'left', 'right']
        blocked = random.choice(directions)

        if blocked != 'top':
            px = self._get_passage_position(x1, wx - 1)
            self.map[wy, px] = 0
        if blocked != 'bottom':
            px = self._get_passage_position(wx + 1, x2)
            self.map[wy, px] = 0
        if blocked != 'left':
            py = self._get_passage_position(y1, wy - 1)
            self.map[py, wx] = 0
        if blocked != 'right':
            py = self._get_passage_position(wy + 1, y2)
            self.map[py, wx] = 0

        # Recurse into four sections
        self._divide(x1, wx - 1, y1, wy - 1)  # Top-left
        self._divide(wx + 1, x2, y1, wy - 1)  # Top-right
        self._divide(x1, wx - 1, wy + 1, y2)  # Bottom-left
        self._divide(wx + 1, x2, wy + 1, y2)  # Bottom-right

    def generate_map(self) -> np.ndarray:
        self.map = np.zeros((self.height, self.width), dtype=np.int8)
        self.map[0, :] = 1
        self.map[-1, :] = 1
        self.map[:, 0] = 1
        self.map[:, -1] = 1

        self._divide(1, self.width - 2, 1, self.height - 2)
        return self.map


class WilsonMazeGenerator(MazeGenerator):
    def generate_map(self) -> np.ndarray:
        # Use a temporary working grid for the algorithm
        # 0 = in maze, 1 = wall, 2 = unvisited
        working_grid = np.full((self.height, self.width), 2, dtype=np.int8)
        working_grid[0, :] = working_grid[-1, :] = 1
        working_grid[:, 0] = working_grid[:, -1] = 1

        unvisited = {(y, x) for y in range(1, self.height - 1, 2)
                             for x in range(1, self.width - 1, 2)}
        start = random.choice(list(unvisited))
        unvisited.remove(start)
        working_grid[start] = 0

        while unvisited:
            cell = random.choice(list(unvisited))
            path = [cell]
            visited_in_walk = {cell}

            while path[-1] in unvisited:
                y, x = path[-1]
                neighbors = [(y + dy*2, x + dx*2) for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]
                             if 1 <= y + dy*2 < self.height - 1 and 1 <= x + dx*2 < self.width - 1]
                next_cell = random.choice(neighbors)

                if next_cell in path:
                    # Erase loop
                    loop_index = path.index(next_cell)
                    path = path[:loop_index + 1]
                else:
                    path.append(next_cell)

            # Carve path into maze
            for i in range(len(path) - 1):
                y1, x1 = path[i]
                y2, x2 = path[i + 1]
                working_grid[y1, x1] = 0
                working_grid[(y1 + y2) // 2, (x1 + x2) // 2] = 0
                unvisited.discard((y1, x1))
                unvisited.discard((y2, x2))

        # Convert the working grid to a binary map (0s and 1s only)
        self.map = np.ones((self.height, self.width), dtype=np.int8)
        self.map[working_grid == 0] = 0  # Set paths to 0
        
        return self.map


class TerrainGenerator(MazeGenerator):
    def __init__(self, width: int, height: int, blob_density: float = 0.002,
                 shape_size: int = 2):
        super().__init__(width, height)
        self.blob_density = blob_density
        self.shape_size = shape_size

    def _place_random_shape(self, terrain: np.ndarray, y: int, x: int):
        for dy in range(-self.shape_size, self.shape_size + 1):
            for dx in range(-self.shape_size, self.shape_size + 1):
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.height and 0 <= nx < self.width:
                    if random.random() < 0.5:
                        terrain[ny, nx] = 1

    def _smooth(self, terrain: np.ndarray, threshold: int = 3) -> np.ndarray:
        padded = np.pad(terrain, 1, mode='constant')
        smoothed = np.zeros_like(terrain)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                smoothed += padded[1 + dy:self.height + 1 + dy, 1 + dx:self.width + 1 + dx]

        return (smoothed >= threshold).astype(np.int8)

    def generate_map(self) -> np.ndarray:
        terrain = np.zeros((self.height, self.width), dtype=np.int8)
        num_seeds = int(self.height * self.width * self.blob_density)

        for _ in range(num_seeds):
            y = random.randint(0, self.height - 1)
            x = random.randint(0, self.width - 1)
            self._place_random_shape(terrain, y, x)

        terrain = self._smooth(terrain)
        # Add border of 1s
        terrain[0, :] = terrain[-1, :] = 1
        terrain[:, 0] = terrain[:, -1] = 1
        return terrain


class GaussianNoiseGenerator(MazeGenerator):
    def __init__(self, width: int, height: int, threshold: float = 0.65, 
                 sigma_x: float = 1.0, sigma_y: float = 1.0, seed: int = None):
        """
        Initialize a Gaussian noise maze generator.
        
        Args:
            width (int): Width of the maze
            height (int): Height of the maze
            threshold (float): Threshold value (0.0-1.0) above which cells become obstacles
            sigma_x (float): Standard deviation of the Gaussian kernel in x direction
            sigma_y (float): Standard deviation of the Gaussian kernel in y direction
            seed (int, optional): Random seed for reproducibility
        """
        super().__init__(width, height)
        self.threshold = threshold
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.seed = seed
        
    def generate_map(self) -> np.ndarray:
        """
        Generate a maze using 2D Gaussian noise.
        
        Returns:
            np.ndarray: Generated maze with borders (1s) and empty spaces (0s)
        """
        # Set random seed if provided
        if self.seed is not None:
            np.random.seed(self.seed)
            
        # Generate random noise
        noise = np.random.random((self.height, self.width))
        
        # Apply Gaussian filter to the noise to create smoother patterns
        smooth_noise = gaussian_filter(noise, sigma=(self.sigma_y, self.sigma_x))
        
        # Normalize to 0-1 range
        smooth_noise = (smooth_noise - smooth_noise.min()) / (smooth_noise.max() - smooth_noise.min())
        
        # Apply threshold to create binary map (0 = path, 1 = wall)
        self.map = (smooth_noise > self.threshold).astype(np.int8)
        
        # Ensure the border is all walls
        self.map[0, :] = 1  # Top edge
        self.map[-1, :] = 1  # Bottom edge
        self.map[:, 0] = 1  # Left edge
        self.map[:, -1] = 1  # Right edge
        
        return self.map


def generate_maze_dataset(num_mazes: int, width: int, height: int, generator_type: str = 'prim') -> List[np.ndarray]:
    """
    Generate a dataset of random mazes.
    
    Args:
        num_mazes (int): Number of mazes to generate
        width (int): Width of each maze
        height (int): Height of each maze
        generator_type (str): Type of maze generator to use ('prim', 'recursive', 'wilson', 'terrain', 'gaussian')
        
    Returns:
        List[np.ndarray]: List of generated mazes
    """
    if generator_type.lower() == 'prim':
        generator = PrimMazeGenerator(width, height)
    elif generator_type.lower() == 'recursive':
        generator = RecursiveDivisionMazeGenerator(width, height)
    elif generator_type.lower() == 'wilson':
        generator = WilsonMazeGenerator(width, height)
    elif generator_type.lower() == 'terrain':
        generator = TerrainGenerator(width, height)
    elif generator_type.lower() == 'gaussian':
        generator = GaussianNoiseGenerator(width, height)
    else:
        raise ValueError("generator_type must be one of 'prim', 'recursive', 'wilson', 'terrain', or 'gaussian'")
    
    return [generator.generate_map() for _ in range(num_mazes)]

def plot_mazes(mazes: Union[np.ndarray, List[np.ndarray]], figsize: Tuple[int, int] = (10, 10)):
    """
    Plot one or multiple mazes in a grid layout.
    
    Args:
        mazes (Union[np.ndarray, List[np.ndarray]]): Single maze or list of mazes to plot
        figsize (Tuple[int, int]): Figure size in inches (width, height)
    """
    if isinstance(mazes, np.ndarray):
        mazes = [mazes]
    
    n_mazes = len(mazes)
    grid_size = int(np.ceil(np.sqrt(n_mazes)))
    
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    if n_mazes == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Define colors for different cell types
    colors = {
        0: 'white',    # Empty space
        1: 'black',    # Wall
        2: 'black'     # Treat old values as walls for backward compatibility
    }
    
    for idx, maze in enumerate(mazes):
        if idx < len(axes):
            ax = axes[idx]
            # Create a colored grid
            colored_maze = np.zeros((*maze.shape, 3))
            for value, color in colors.items():
                mask = maze == value
                colored_maze[mask] = plt.matplotlib.colors.to_rgb(color)
            
            ax.imshow(colored_maze)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f'Maze {idx + 1}')
    
    # Hide empty subplots
    for idx in range(n_mazes, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

if __name__ == "__main__":
    # Find the absolute path to the configuration file
    script_dir = Path(__file__).parent.absolute()
    config_path = script_dir / 'dataset_config.yaml'
    
    print(f"Script directory: {script_dir}")
    print(f"Looking for config at: {config_path}")
    
    if not config_path.exists():
        print("Config file not found at the expected location.")
        print("Creating a minimal test configuration...")
        
        # Create a minimal test config
        test_config = {
            'name': 'test_dataset',
            'generators': {
                'prim': {
                    'count_per_config': 2,
                    'width': [21],
                    'height': [21]
                }
            }
        }
        
        # Write the test config to disk
        with config_path.open('w') as f:
            yaml.dump(test_config, f, default_flow_style=False)
        print(f"Created test configuration at {config_path}")
    
    # Generate dataset using the configuration file
    dataset_gen = DatasetGenerator(config_path)
    
    # Generate a smaller dataset for testing
    print("\nGenerating a small test dataset...")
    maps = dataset_gen.generate_dataset()
    
    print(f"\nGenerated {len(maps)} maps. Displaying samples...")
    # Plot a few samples
    sample_size = min(4, len(maps))
    sample_indices = random.sample(range(len(maps)), sample_size)
    samples = [maps[i] for i in sample_indices]
    
    fig = plot_mazes(samples, figsize=(10, 10))
    plt.savefig("sample_preview.png")
    print("Saved sample preview to sample_preview.png")
    plt.close(fig)
    
    # Save the dataset with proper organization
    print("\nSaving the dataset...")
    dataset_gen.save_dataset()
    
    print("\nDataset generation and saving completed successfully!")
