import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from collections import deque


class Dataset:
    def __init__(self, dataset_name: str = None, config_file: str = None):
        """
        Initialize a dataset object. Can be used to either load an existing dataset or start tracking a new one.
        
        Args:
            dataset_name (str, optional): Name of the dataset
            config_file (str, optional): Path to a config file if creating a new dataset
        """
        self.dataset_name = dataset_name
        self.config_file = config_file
        self.maps = []
        self.metadata = []
        self.dataset_info = {}
        self.config = {}
    
    def load(self, dataset_path: str, load_maps: bool = True) -> 'Dataset':
        """
        Load a dataset from a saved folder.
        
        Args:
            dataset_path (str): Path to the dataset folder
            load_maps (bool, optional): Whether to load the actual map data or just metadata. Defaults to True.
            
        Returns:
            Dataset: Self for method chaining
        """
        path = Path(dataset_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Dataset path not found: {path}")
        
        # Load metadata.json file
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        
        with metadata_path.open('r') as f:
            metadata_content = json.load(f)
        
        # Set dataset information
        self.dataset_name = metadata_content.get('dataset_name', 'unknown')
        self.dataset_info = {
            'creation_time': metadata_content.get('creation_time', ''),
            'path': str(path.absolute())
        }
        
        # Set configuration
        self.config = metadata_content.get('config', {})
        
        # Load map metadata
        self.metadata = metadata_content.get('maps', [])
        
        # Load actual map data if requested
        if load_maps:
            print(f"Loading {len(self.metadata)} maps from {path}...")
            self.maps = []
            
            for map_meta in tqdm(self.metadata, desc="Loading maps"):
                map_file = map_meta.get('file_path', '')
                if map_file:
                    map_path = path / map_file
                    if map_path.exists():
                        try:
                            map_data = np.load(map_path)
                            self.maps.append(map_data)
                        except Exception as e:
                            print(f"Error loading map {map_path}: {e}")
                            # Add a placeholder for failed loads to maintain index alignment
                            self.maps.append(None)
                    else:
                        print(f"Map file not found: {map_path}")
                        self.maps.append(None)
            
            print(f"Successfully loaded {sum(1 for m in self.maps if m is not None)} maps")
        
        return self
    
    def filter(self, criteria: Dict[str, Any]) -> 'Dataset':
        """
        Filter the dataset based on criteria.
        
        Args:
            criteria (Dict[str, Any]): Criteria to filter by, e.g. {'generator_type': 'prim', 'width': 21}
            
        Returns:
            Dataset: A new Dataset object with filtered maps and metadata
        """
        if not self.metadata:
            print("No metadata available for filtering")
            return self
        
        # Create a new dataset with the same metadata
        filtered_dataset = Dataset(self.dataset_name)
        filtered_dataset.dataset_info = self.dataset_info.copy()
        filtered_dataset.config = self.config.copy()
        
        # Filter metadata and maps
        filtered_indices = []
        
        for i, meta in enumerate(self.metadata):
            match = True
            for key, value in criteria.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
            
            if match:
                filtered_indices.append(i)
        
        # Copy matching metadata
        filtered_dataset.metadata = [self.metadata[i] for i in filtered_indices]
        
        # Copy matching maps if they're loaded
        if self.maps:
            filtered_dataset.maps = [self.maps[i] for i in filtered_indices]
        
        print(f"Filtered {len(self.metadata)} maps to {len(filtered_dataset.metadata)} maps")
        return filtered_dataset
    
    def get_map(self, index: int) -> Optional[np.ndarray]:
        """
        Get a specific map by index.
        
        Args:
            index (int): Index of the map to retrieve
            
        Returns:
            Optional[np.ndarray]: The map data, or None if not available
        """
        if not 0 <= index < len(self.maps):
            print(f"Index {index} out of range (0-{len(self.maps)-1})")
            return None
        
        return self.maps[index]
    
    def get_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific map by index.
        
        Args:
            index (int): Index of the map's metadata to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: The map metadata, or None if not available
        """
        if not 0 <= index < len(self.metadata):
            print(f"Index {index} out of range (0-{len(self.metadata)-1})")
            return None
        
        return self.metadata[index]
    
    def visualize(self, indices: List[int] = None, max_maps: int = 16, figsize: Tuple[int, int] = (15, 15)) -> None:
        """
        Visualize maps from the dataset.
        
        Args:
            indices (List[int], optional): Indices of maps to visualize. If None, randomly selects maps.
            max_maps (int, optional): Maximum number of maps to visualize. Defaults to 16.
            figsize (Tuple[int, int], optional): Figure size. Defaults to (15, 15).
        """
        if not self.maps:
            print("No maps loaded to visualize")
            return
        
        # Select indices if not provided
        if indices is None:
            available_indices = list(range(len(self.maps)))
            selected_count = min(max_maps, len(available_indices))
            indices = sorted(random.sample(available_indices, selected_count))
        else:
            # Filter out invalid indices
            indices = [i for i in indices if 0 <= i < len(self.maps)]
            if not indices:
                print("No valid indices to visualize")
                return
            
            # Limit to max_maps
            indices = indices[:max_maps]
        
        # Get maps to visualize
        maps_to_visualize = [self.maps[i] for i in indices if self.maps[i] is not None]
        if not maps_to_visualize:
            print("No valid maps to visualize")
            return
        
        # Determine grid size
        n_maps = len(maps_to_visualize)
        grid_size = int(np.ceil(np.sqrt(n_maps)))
        
        # Create figure and axes
        fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
        if n_maps == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Define colors for different cell types
        colors = {
            0: 'white',    # Empty space
            1: 'black',    # Wall
            2: 'black'     # Treat old values as walls for backward compatibility
        }
        
        # Plot each map
        for idx, map_data in enumerate(maps_to_visualize):
            if idx < len(axes):
                ax = axes[idx]
                # Create a colored grid
                colored_map = np.zeros((*map_data.shape, 3))
                for value, color in colors.items():
                    mask = map_data == value
                    colored_map[mask] = plt.matplotlib.colors.to_rgb(color)
                
                ax.imshow(colored_map)
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add metadata as title if available
                if indices[idx] < len(self.metadata):
                    meta = self.metadata[indices[idx]]
                    generator = meta.get('generator_type', 'unknown')
                    dims = f"{meta.get('width', '?')}x{meta.get('height', '?')}"
                    ax.set_title(f"Map {indices[idx]}\n{generator} {dims}")
                else:
                    ax.set_title(f"Map {indices[idx]}")
        
        # Hide empty subplots
        for idx in range(n_maps, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def summary(self) -> None:
        """
        Print a summary of the dataset
        """
        print(f"Dataset: {self.dataset_name}")
        print(f"Location: {self.dataset_info.get('path', 'N/A')}")
        print(f"Creation time: {self.dataset_info.get('creation_time', 'Unknown')}")
        print(f"Maps: {len(self.metadata)} metadata entries, {len(self.maps)} loaded maps")
        
        # Count maps by generator type
        if self.metadata:
            generator_counts = {}
            for meta in self.metadata:
                gen_type = meta.get('generator_type', 'unknown')
                generator_counts[gen_type] = generator_counts.get(gen_type, 0) + 1
            
            print("\nGenerator types:")
            for gen_type, count in generator_counts.items():
                print(f"  - {gen_type}: {count} maps")
        
        # Show unique dimensions
        if self.metadata:
            dimensions = set()
            for meta in self.metadata:
                width = meta.get('width', 0)
                height = meta.get('height', 0)
                dimensions.add(f"{width}x{height}")
            
            print("\nDimensions:")
            for dim in sorted(dimensions):
                print(f"  - {dim}")
    
    def sample(self, 
               generator_type: str = None, 
               width: int = None, 
               height: int = None,
               random_transform: bool = False,
               filter_criteria: Dict[str, Any] = None,
               add_endpoint: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Sample a random map from the dataset, optionally filtered by criteria and randomly transformed.
        
        Args:
            generator_type (str, optional): Type of generator to sample from ('prim', 'recursive', etc.)
            width (int, optional): Width of map to sample
            height (int, optional): Height of map to sample
            random_transform (bool, optional): Whether to apply random rotation and flipping
            filter_criteria (Dict[str, Any], optional): Additional filter criteria (same as used in filter method)
            add_endpoint (bool, optional): Whether to add a random endpoint (target cell with value 3) to the map
            
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: The sampled map and its metadata
        """
        if not self.maps:
            raise ValueError("No maps loaded in the dataset")
        
        # Build filter criteria
        criteria = {}
        if generator_type is not None:
            criteria['generator_type'] = generator_type
        if width is not None:
            criteria['width'] = width
        if height is not None:
            criteria['height'] = height
            
        # Add additional filter criteria if provided
        if filter_criteria is not None:
            criteria.update(filter_criteria)
        
        # Find eligible indices
        eligible_indices = []
        for i, meta in enumerate(self.metadata):
            if i >= len(self.maps) or self.maps[i] is None:
                continue
                
            match = True
            for key, value in criteria.items():
                if key not in meta or meta[key] != value:
                    match = False
                    break
            
            if match:
                eligible_indices.append(i)
        
        if not eligible_indices:
            criteria_str = ", ".join(f"{k}={v}" for k, v in criteria.items())
            raise ValueError(f"No maps found matching criteria: {criteria_str}")
        
        # Select a random map
        idx = random.choice(eligible_indices)
        map_data = self.maps[idx].copy()  # Copy to avoid modifying the original
        metadata = self.metadata[idx].copy()
        
        # Apply random transformations
        if random_transform:
            # Random number of 90-degree rotations (0, 1, 2, or 3)
            rotations = random.randint(0, 3)
            if rotations > 0:
                map_data = np.rot90(map_data, rotations)
                metadata['transform'] = f'rotation_{rotations * 90}deg'
            
            # Random flip (50% chance)
            if random.random() < 0.5:
                map_data = np.fliplr(map_data)
                metadata['transform'] = metadata.get('transform', '') + '_flipped'
        
        # Add a random endpoint (target cell) if requested
        if add_endpoint:
            # Find all empty cells (value=0)
            empty_cells = np.where(map_data == 0)
            if len(empty_cells[0]) > 0:
                # Randomly select one empty cell
                random_idx = random.randint(0, len(empty_cells[0]) - 1)
                endpoint_y, endpoint_x = empty_cells[0][random_idx], empty_cells[1][random_idx]
                
                # Set the selected cell as a target (value=3)
                map_data[endpoint_y, endpoint_x] = 3
                
                # Add endpoint information to metadata
                metadata['endpoint'] = (int(endpoint_y), int(endpoint_x))
            else:
                print("Warning: No empty cells found for endpoint placement")
        
        return map_data, metadata
    
    def sample_batch(self, 
                    batch_size: int, 
                    generator_type: str = None,
                    width: int = None, 
                    height: int = None,
                    random_transform: bool = False,
                    filter_criteria: Dict[str, Any] = None,
                    add_endpoint: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Sample a batch of maps from the dataset.
        
        Args:
            batch_size (int): Number of maps to sample
            generator_type (str, optional): Type of generator to sample from
            width (int, optional): Width of maps to sample
            height (int, optional): Height of maps to sample
            random_transform (bool, optional): Whether to apply random transformations
            filter_criteria (Dict[str, Any], optional): Additional filter criteria
            add_endpoint (bool, optional): Whether to add a random endpoint (target cell with value 3) to each map
            
        Returns:
            Tuple[np.ndarray, List[Dict[str, Any]]]: Batch of maps and their metadata
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        maps = []
        metadata_list = []
        
        for _ in range(batch_size):
            map_data, metadata = self.sample(
                generator_type=generator_type, 
                width=width, 
                height=height, 
                random_transform=random_transform,
                filter_criteria=filter_criteria,
                add_endpoint=add_endpoint
            )
            maps.append(map_data)
            metadata_list.append(metadata)
        
        # Try to stack maps if they have the same shape
        try:
            maps_array = np.stack(maps)
        except ValueError:
            # Maps have different shapes, return as list
            return maps, metadata_list
        
        return maps_array, metadata_list
    
    def get_generator_types(self) -> List[str]:
        """
        Get the list of unique generator types in the dataset.
        
        Returns:
            List[str]: List of generator types
        """
        if not self.metadata:
            return []
        
        generator_types = set()
        for meta in self.metadata:
            gen_type = meta.get('generator_type', 'unknown')
            generator_types.add(gen_type)
        
        return sorted(list(generator_types))
    
    def get_dimensions(self) -> List[Tuple[int, int]]:
        """
        Get the list of unique dimensions in the dataset.
        
        Returns:
            List[Tuple[int, int]]: List of (width, height) tuples
        """
        if not self.metadata:
            return []
        
        dimensions = set()
        for meta in self.metadata:
            width = meta.get('width', 0)
            height = meta.get('height', 0)
            dimensions.add((width, height))
        
        return sorted(list(dimensions))
    
    def ensure_connectivity(self, index: int = None) -> int:
        """
        Ensures that all empty spaces in a map form a single connected component.
        This is done by removing obstacles (wall cells) until all empty spaces are connected.
        
        Args:
            index (int, optional): Index of the map to process. If None, processes all maps.
            
        Returns:
            int: Total number of obstacles removed across all processed maps
        """
        if not self.maps:
            print("No maps loaded")
            return 0
            
        if index is not None:
            # Process a single map
            if 0 <= index < len(self.maps) and self.maps[index] is not None:
                removed = self._ensure_single_map_connectivity(self.maps[index])
                if removed > 0:
                    print(f"Removed {removed} obstacles from map {index}")
                return removed
            else:
                print(f"Invalid map index {index}")
                return 0
        else:
            # Process all maps
            total_removed = 0
            print(f"Ensuring connectivity for {len(self.maps)} maps...")
            for i, maze in enumerate(self.maps):
                if maze is not None:
                    removed = self._ensure_single_map_connectivity(maze)
                    total_removed += removed
                    if removed > 0:
                        print(f"Removed {removed} obstacles from map {i}")
            
            print(f"Total obstacles removed: {total_removed}")
            return total_removed
        
    def _ensure_single_map_connectivity(self, maze: np.ndarray) -> int:
        """
        Ensures connectivity in a single map by removing obstacles until 
        all empty spaces form a single connected component.
        Only removes the minimum number of obstacles necessary.
        Preserves all border walls by excluding them from the connectivity process.
        
        Args:
            maze (np.ndarray): The maze to process
            
        Returns:
            int: Number of obstacles removed
        """
        height, width = maze.shape
        removed_count = 0
        
        # If the maze is too small to have an inner area, return
        if height <= 2 or width <= 2:
            return 0
        
        # Create a working copy of the maze excluding the border
        inner_maze = maze[1:-1, 1:-1].copy()
        
        # Get all components in the inner maze
        components, total_components = self._get_connected_components(inner_maze)
        
        # If already connected, nothing to do
        if total_components <= 1:
            return 0
        
        # Sort components by size (largest first)
        components.sort(key=len, reverse=True)
        
        # Get the largest component
        merged_component = components[0].copy()
        
        # Create a set of all obstacles adjacent to the merged component
        adjacent_obstacles = set()
        for y, x in merged_component:
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < height-2 and 0 <= nx < width-2 and 
                    inner_maze[ny, nx] == 1):  # It's a wall
                    adjacent_obstacles.add((ny, nx))
        
        # Process smaller components one by one
        for component in components[1:]:
            # Find obstacles adjacent to this component
            component_obstacles = set()
            for y, x in component:
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < height-2 and 0 <= nx < width-2 and 
                        inner_maze[ny, nx] == 1):  # It's a wall
                        component_obstacles.add((ny, nx))
            
            # Find the best obstacle to remove (one that is adjacent to both components)
            common_obstacles = adjacent_obstacles.intersection(component_obstacles)
            
            if common_obstacles:
                # We have a direct connection - pick any common obstacle
                obstacle = next(iter(common_obstacles))
                
                # Remove the obstacle
                y, x = obstacle
                inner_maze[y, x] = 0
                removed_count += 1
                
                # Update merged component and adjacent obstacles
                merged_component.add(obstacle)
                merged_component.update(component)
                
                # Update adjacent obstacles (remove the one we just cleared)
                adjacent_obstacles.remove(obstacle)
                
                # Add new obstacles that may be adjacent to our expanded merged component
                for cy, cx in component:
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < height-2 and 0 <= nx < width-2 and 
                            inner_maze[ny, nx] == 1):  # It's a wall
                            adjacent_obstacles.add((ny, nx))
            else:
                # No direct connection found, find the shortest path
                min_distance = float('inf')
                best_pair = None
                
                # For each cell in the component, find the closest cell in the merged component
                for cy1, cx1 in component:
                    for cy2, cx2 in merged_component:
                        distance = abs(cy1 - cy2) + abs(cx1 - cx2)
                        if distance < min_distance:
                            min_distance = distance
                            best_pair = ((cy1, cx1), (cy2, cx2))
                
                if best_pair:
                    # Create a path between the cells
                    (y1, x1), (y2, x2) = best_pair
                    
                    # Try to create a shorter path by moving through a common wall
                    if abs(y1 - y2) == 2 and x1 == x2:  # Cells separated by one wall vertically
                        wall_y = (y1 + y2) // 2
                        wall_x = x1
                        if inner_maze[wall_y, wall_x] == 1:
                            inner_maze[wall_y, wall_x] = 0
                            removed_count += 1
                            merged_component.add((wall_y, wall_x))
                            merged_component.add((y1, x1))
                            merged_component.update(component)
                            continue
                            
                    elif abs(x1 - x2) == 2 and y1 == y2:  # Cells separated by one wall horizontally
                        wall_y = y1
                        wall_x = (x1 + x2) // 2
                        if inner_maze[wall_y, wall_x] == 1:
                            inner_maze[wall_y, wall_x] = 0
                            removed_count += 1
                            merged_component.add((wall_y, wall_x))
                            merged_component.add((y1, x1))
                            merged_component.update(component)
                            continue
                    
                    # Otherwise, create an L-shaped path (horizontal first)
                    current_y, current_x = y1, x1
                    
                    # Add the starting cell to the merged component
                    merged_component.add((current_y, current_x))
                    
                    # Move horizontally first
                    while current_x != x2:
                        current_x += 1 if current_x < x2 else -1
                        if inner_maze[current_y, current_x] == 1:  # It's a wall
                            inner_maze[current_y, current_x] = 0
                            removed_count += 1
                        merged_component.add((current_y, current_x))
                    
                    # Then move vertically
                    while current_y != y2:
                        current_y += 1 if current_y < y2 else -1
                        if inner_maze[current_y, current_x] == 1:  # It's a wall
                            inner_maze[current_y, current_x] = 0
                            removed_count += 1
                        merged_component.add((current_y, current_x))
                    
                    # Update the merged component with all cells from the component
                    merged_component.update(component)
                    
                    # Update adjacent obstacles for the newly expanded merged component
                    adjacent_obstacles = set()
                    for y, x in merged_component:
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = y + dy, x + dx
                            if (0 <= ny < height-2 and 0 <= nx < width-2 and 
                                inner_maze[ny, nx] == 1):  # It's a wall
                                adjacent_obstacles.add((ny, nx))
        
        # Verify that all empty spaces are now connected
        new_components, total_new_components = self._get_connected_components(inner_maze)
        
        if total_new_components > 1:
            # If we still have multiple components, we need to be more aggressive
            # Connect all remaining components to the largest one
            merged_component = max(new_components, key=len)
            other_components = [c for c in new_components if c != merged_component]
            
            for component in other_components:
                # Find the closest pair between this component and the merged component
                min_distance = float('inf')
                best_pair = None
                
                for cy1, cx1 in component:
                    for cy2, cx2 in merged_component:
                        distance = abs(cy1 - cy2) + abs(cx1 - cx2)
                        if distance < min_distance:
                            min_distance = distance
                            best_pair = ((cy1, cx1), (cy2, cx2))
                
                if best_pair:
                    # Create a direct L-shaped path
                    (y1, x1), (y2, x2) = best_pair
                    current_y, current_x = y1, x1
                    
                    # Move horizontally first
                    while current_x != x2:
                        current_x += 1 if current_x < x2 else -1
                        if inner_maze[current_y, current_x] == 1:  # It's a wall
                            inner_maze[current_y, current_x] = 0
                            removed_count += 1
                    
                    # Then move vertically
                    while current_y != y2:
                        current_y += 1 if current_y < y2 else -1
                        if inner_maze[current_y, current_x] == 1:  # It's a wall
                            inner_maze[current_y, current_x] = 0
                            removed_count += 1
                    
                    # Update merged component with connected component
                    merged_component.update(component)
            
            # Final verification
            _, total_final_components = self._get_connected_components(inner_maze)
            if total_final_components > 1:
                print(f"Warning: Still have {total_final_components} components after aggressive connection.")
        
        # Apply changes from inner_maze back to the original maze
        maze[1:-1, 1:-1] = inner_maze
        
        return removed_count
    
    def _get_connected_components(self, maze):
        """
        Helper method to find all connected components in a maze.
        
        Args:
            maze (np.ndarray): The maze to analyze
            
        Returns:
            tuple: (list of components, number of components)
                   Each component is a set of (y, x) coordinates
        """
        from collections import deque
        
        height, width = maze.shape
        visited = set()
        components = []
        
        for y in range(height):
            for x in range(width):
                if maze[y, x] == 0 and (y, x) not in visited:
                    # Found a new component
                    component = set()
                    queue = deque([(y, x)])
                    visited.add((y, x))
                    
                    # BFS to find all cells in this component
                    while queue:
                        cy, cx = queue.popleft()
                        component.add((cy, cx))
                        
                        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            ny, nx = cy + dy, cx + dx
                            neighbor = (ny, nx)
                            
                            if (0 <= ny < height and 0 <= nx < width and 
                                maze[ny, nx] == 0 and neighbor not in visited):
                                queue.append(neighbor)
                                visited.add(neighbor)
                    
                    components.append(component)
        
        return components, len(components)
