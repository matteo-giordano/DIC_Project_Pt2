from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Maze:
    def __init__(self, array: np.ndarray, step_size=0.4):
        if not self._validate_array(array):
            raise ValueError("Invalid array")
        self.array = array.copy()
        self.map_height, self.map_width = self.array.shape

        self.action_map = {
            0: np.array([1., 0.]),    # North
            1: np.array([1., 1.]),    # North-East
            2: np.array([0., 1.]),    # East
            3: np.array([-1., 1.]),   # South-East
            4: np.array([-1., 0.]),   # South
            5: np.array([-1., -1.]),  # South-West
            6: np.array([0., -1.]),   # West
            7: np.array([1., -1.]),   # North-West
        }

        self.step_size = step_size
        self.goal_radius = 1.
        self.agent_radius = 0.15

        # Agent & Goal positions stored as numpy arrays (-(y+1), x) , so agent_pos = (1.5, 1.5) -> array[-2, 1]
        self.agent_pos = np.array([1.5, 1.5])
        self.goal_pos = np.array(self.array.shape) - self.agent_pos  

        # Rendering attributes
        self.fig = None
        self.ax = None
        self.agent_circle = None
        self.goal_circle = None

    def step(self, action: int):
        # Get action & compute new position
        action_vec = self.action_map[action] * self.step_size
        new_pos = self.agent_pos + action_vec

        # Check collision with walls
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            # Try to slide along walls
            self.agent_pos = self._slide_along_wall(action_vec)

    def reset(self):
        pass

    def render(self):
        """Render the maze with agent and target. Initialize plot on first call."""
        if self.fig is None:
            # Initialize the plot
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            
            # Display the maze (flip vertically to match coordinate system)
            # Use inverted gray colormap so 0's are white and 1's are black
            self.ax.imshow(np.flipud(self.array), cmap='gray_r', origin='lower', extent=[0, self.map_width, 0, self.map_height])
            
            # Create agent circle (blue)
            self.agent_circle = patches.Circle((self.agent_pos[1], self.agent_pos[0]), 
                                             self.agent_radius, color='blue', alpha=0.8)
            self.ax.add_patch(self.agent_circle)
            
            # Create goal circle (red)
            self.goal_circle = patches.Circle((self.goal_pos[1], self.goal_pos[0]), 
                                            self.goal_radius, color='red', alpha=0.6)
            self.ax.add_patch(self.goal_circle)
            
            # Set up the plot
            self.ax.set_xlim(0, self.map_width)
            self.ax.set_ylim(0, self.map_height)
            self.ax.set_aspect('equal')
            self.ax.set_title('Maze Environment')
            self.ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
        else:
            # Update existing plot
            # Update agent position (note: matplotlib uses (x, y) while we use (y, x))
            self.agent_circle.center = (self.agent_pos[1], self.agent_pos[0])
            
            # Update goal position (in case it changed)
            self.goal_circle.center = (self.goal_pos[1], self.goal_pos[0])
        
        # Refresh the display
        plt.draw()
        plt.pause(0.01)  # Small pause to allow plot to update

    def _validate_array(self, array: np.ndarray):
        assert array.ndim == 2, "Array must be 2D"
        assert array.shape[0] > 0 and array.shape[1] > 0, "Array must have non-zero dimensions"
        assert array.dtype == np.int8, "Array must be of type int8"
        assert array.min() >= 0 and array.max() <= 3, "Array values must be between 0 and 3"
        assert array.shape[0] % 2 == 1 and array.shape[1] % 2 == 1, "Array dimensions must be odd"
        assert np.all(array[0, :] == 1) and np.all(array[-1, :] == 1), "Array must have boundaries"
        assert np.all(array[:, 0] == 1) and np.all(array[:, -1] == 1), "Array must have boundaries"
        return True

    def _is_valid_position(self, position: np.ndarray) -> bool:
            """Check if the agent at given position would collide with walls."""
            y, x = position
            num_samples = 16
            angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
            
            for angle in angles:
                # Calculate point on agent's circumference
                sample_y = y + self.agent_radius * np.sin(angle)
                sample_x = x + self.agent_radius * np.cos(angle)
                # plt.plot(sample_x, sample_y, 'ro')
                
                if self.array[-int(np.floor(sample_y))- 1, int(np.floor(sample_x))] == 1:
                    print(f"Collision at {sample_y}, {sample_x}")
                    return False
            print(f"No collision at {y}, {x}, {int(np.floor(sample_y))}, {int(np.floor(sample_x))}")
            return True

    def _slide_along_wall(self, action_vec: np.ndarray) -> np.ndarray:
        """Move as far as possible in the desired direction until hitting a wall."""
        current_pos = self.agent_pos
        
        # Normalize the action vector to get direction
        if np.linalg.norm(action_vec) == 0:
            return current_pos
            
        direction = action_vec / np.linalg.norm(action_vec)
        
        # Binary search to find the maximum distance we can move
        min_distance = 0.0
        max_distance = np.linalg.norm(action_vec)
        epsilon = 0.01  # Precision for binary search
        
        best_pos = current_pos
        
        while max_distance - min_distance > epsilon:
            mid_distance = (min_distance + max_distance) / 2.0
            test_pos = current_pos + direction * mid_distance
            
            if self._is_valid_position(test_pos):
                # Can move this far, try going further
                min_distance = mid_distance
                best_pos = test_pos
            else:
                # Can't move this far, reduce distance
                max_distance = mid_distance
        
        return best_pos


class Environment:
    def __init__(self, maze: Maze):
        self.maze = maze

    def step(self, action: int):
        self.maze.step(action)
        return self._get_observation(), self.is_done()

    def _get_observation(self):
        return np.concatenate([self.maze.agent_pos, self.maze.goal_pos])

    def reset(self):
        self.maze.agent_pos = np.array([1.5, 1.5]) + np.random.normal(0, 0.2, size=self.maze.agent_pos.shape)
        return self._get_observation()

    def render(self):
        self.maze.render()

    def is_done(self):
        return np.linalg.norm(self.maze.agent_pos - self.maze.goal_pos) < self.maze.goal_radius

if __name__ == "__main__":
    from time import sleep
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Maze(warehouse_map)
    env.agent_pos = np.array([1.5, 22.5])
    for _ in range(100):
        env.render()
        action = np.random.randint(0, 8)
        env.step(action)
        sleep(0.01)
                                                        