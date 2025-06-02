from shapely.geometry import Point, Polygon
import networkx as nx
import numpy as np
from collections import defaultdict
import math

class ContinuousWorld:
    def __init__(self, width: float, height: float, obstacles: list[Polygon], start: tuple[float, float], goal: tuple[float, float]):
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.start = Point(start)
        self.goal = Point(goal)
        self.agent = self.start
        self.visit_count = defaultdict(int)
        self.graph = self.build_graph()
        self.start_cell = (self.start.x, self.start.y)       
        self.target_cell = (self.goal.x, self.goal.y)

    def is_valid(self, point: Point) -> bool:
        """Check if the point is inside bounds and not in any obstacle."""
        if not (0 <= point.x <= self.width and 0 <= point.y <= self.height):
            return False
        for obstacle in self.obstacles:
            if point.within(obstacle):
                return False
        return True

    def move_agent(self, new_pos: tuple[float, float]):
        """Moves the agent to a new position if it's valid."""
        point = Point(new_pos)
        if not self.is_valid(point):
            raise ValueError("Move would place agent in obstacle or out of bounds.")
        self.agent = point
        self.visit_count[(point.x, point.y)] += 1

    def is_done(self) -> bool:
        return self.agent.distance(self.goal) < 1.0  # within 1 unit is close enough

    def reset(self):
        self.agent = self.start
        self.visit_count = defaultdict(int)
        return self.agent

    def build_graph(self):
        """Proximity-based graph with evenly distributed 360-degree sampling from start and goal."""
        G = nx.Graph()
        sample_points = [self.start, self.goal]

        # Parameters
        total_samples = 2000
        num_directions = 360
        samples_per_direction = total_samples / num_directions  # ≈1.39
        steps_per_ray = int(np.ceil(samples_per_direction))
        max_distance = min(self.width, self.height) / 2  # max radius
        step_size = max_distance / steps_per_ray

        # Sample points in 360 directions from both start and goal
        for origin in [self.start, self.goal]:
            for degree in range(num_directions):
                angle = math.radians(degree)
                for step in range(1, steps_per_ray + 1):
                    dx = math.cos(angle) * step_size * step
                    dy = math.sin(angle) * step_size * step
                    p = Point(origin.x + dx, origin.y + dy)
                    if 0 <= p.x <= self.width and 0 <= p.y <= self.height and self.is_valid(p):
                        sample_points.append(p)

        # Add edges using visibility (line-of-sight)
        for i, p1 in enumerate(sample_points):
            for j in range(i + 1, len(sample_points)):
                p2 = sample_points[j]
                line = p1.buffer(0.1).union(p2.buffer(0.1)).convex_hull
                if all(not line.intersects(obs) for obs in self.obstacles):
                    G.add_edge((p1.x, p1.y), (p2.x, p2.y), weight=p1.distance(p2))

        return G
    
    def __str__(self):
        return f"ContinuousWorld({self.width}x{self.height}), agent at ({self.agent.x:.2f}, {self.agent.y:.2f}), goal at ({self.goal.x:.2f}, {self.goal.y:.2f}), {len(self.obstacles)} obstacles"