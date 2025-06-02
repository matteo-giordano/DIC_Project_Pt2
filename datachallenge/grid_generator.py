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
from shapely.geometry import Polygon, Point, mapping
from shapely.affinity import translate, scale


class ContinuousMapGenerator(ABC):
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height

    @abstractmethod
    def generate_map(self) -> List[Polygon]:
        """
        Generate a continuous 2D map represented as geometric obstacles.
        
        Returns:
            List[Polygon]: List of polygonal obstacles
        """
        pass


class ProceduralGeometryMapGenerator(ContinuousMapGenerator):
    def __init__(self,
                 width: float,
                 height: float,
                 num_obstacles: int = 20,
                 shape_type: str = 'circle',  # 'circle' or 'polygon'
                 size_range: Tuple[float, float] = (1.0, 5.0)):
        super().__init__(width, height)
        self.num_obstacles = num_obstacles
        self.shape_type = shape_type
        self.size_range = size_range

    def _generate_circle(self, center: Tuple[float, float], radius: float) -> Polygon:
        return Point(center).buffer(radius)

    def _generate_polygon(self, center: Tuple[float, float], scale_factor: float) -> Polygon:
        # Start from a unit triangle and scale/move it
        base = Polygon([(0, 0), (1, 0), (0.5, 1)])
        return scale(translate(base, xoff=center[0], yoff=center[1]), xfact=scale_factor, yfact=scale_factor)

    def generate_map(self) -> List[Polygon]:
        obstacles = []
        for _ in range(self.num_obstacles):
            cx = random.uniform(0, self.width)
            cy = random.uniform(0, self.height)
            size = random.uniform(*self.size_range)

            if self.shape_type == 'circle':
                obstacle = self._generate_circle((cx, cy), size)
            elif self.shape_type == 'polygon':
                obstacle = self._generate_polygon((cx, cy), size)
            else:
                raise ValueError("shape_type must be 'circle' or 'polygon'")

            obstacles.append(obstacle)

        return obstacles


def plot_geometry_map(obstacles: List[Polygon], width: float, height: float, figsize=(10, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    for shape in obstacles:
        x, y = shape.exterior.xy
        ax.fill(x, y, color='black')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.set_title("Continuous Geometric Map")
    plt.tight_layout()
    plt.show()


def save_continuous_maze(obstacles: List[Polygon], width: float, height: float, save_dir: str = "datachallenge/grid_configs", filename: str = None):
    # Default filename with timestamp
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"continuous_maze_{timestamp}.json"

    # Prepare obstacle data in GeoJSON-like format
    maze_data = {
        "type": "ContinuousMaze",
        "width": width,
        "height": height,
        "obstacles": [mapping(ob) for ob in obstacles]
    }

    file_path = Path(save_dir) / filename
    with file_path.open("w") as f:
        json.dump(maze_data, f, indent=2)

    print(f"Saved continuous maze to: {file_path}")

# Example usage
if __name__ == "__main__":
    width, height = 100, 100
    generator = ProceduralGeometryMapGenerator(width, height, shape_type='circle', num_obstacles=50)
    obstacles = generator.generate_map()
    plot_geometry_map(obstacles, width, height)
    save_continuous_maze(obstacles, width, height)