import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Maze:
    def __init__(self, array: np.ndarray, step_size=0.4):
        if not self._validate_array(array):
            raise ValueError("Invalid array")
        self.array = array.copy()    # 2D numpy array representing the maze map
        self.map_height, self.map_width = self.array.shape
        diag = 1 / np.sqrt(2)        # 1 / sqrt(2), for diagonal movement vectors

        # Dictionary mapping actions (0–7) to 2D direction vectors
        self.action_map = { 
            0: np.array([1., 0.]),            # North
            1: np.array([diag, diag]),        # North-East
            2: np.array([0., 1.]),            # East
            3: np.array([-diag, diag]),       # South-East
            4: np.array([-1., 0.]),           # South
            5: np.array([-diag, -diag]),      # South-West
            6: np.array([0., -1.]),           # West
            7: np.array([diag, -diag]),       # North-West
        }
        self.step_size = step_size # Distance moved per step
        self.goal_radius = 1.      # Radius within which the goal is considered reached
        self.agent_radius = 0.15   # Radius of the agent for collision checking

        # Agent & Goal positions stored as numpy arrays (-(y+1), x) , so agent_pos = (1.5, 1.5) -> array[-2, 1]
        self.agent_pos = np.array([1.5, 22.5])                            # Agent's current position (float coordinates)
        self.goal_pos = np.array(self.array.shape) - np.array([2.5, 2.5]) # Goal position (float coordinates)

        # Rendering attributes
        self.fig = None
        self.ax = None
        self.agent_circle = None
        self.goal_circle = None

        # Pre-compute action vectors for efficiency
        self.action_vectors = np.array([self.action_map[i] for i in range(8)]) * self.step_size # Precomputed movement vectors for each action, scaled by self.step_size

    def step(self, action: int):
        """
        Move the agent one step in the specified direction.
        If the movement is valid (no wall collision), update the position.
        Otherwise, slide along the wall as far as possible in the intended direction.
        Args:
            action (int): An integer in [0, 7] indicating one of the 8 compass directions.
        """
        # Get action & compute new position
        action_vec = self.action_vectors[action]
        new_pos = self.agent_pos + action_vec
        # Check collision with walls
        if self._is_valid_position(new_pos):
            self.agent_pos = new_pos
        else:
            # Try to slide along walls
            self.agent_pos = self._slide_along_wall(action_vec)

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
        """
        Check if the agent at given position would collide with walls.
        Uses circular sampling around the agent's radius.
        Args:
            position (np.ndarray): The position to validate.
        Returns:
            bool: True if the position is collision-free, False otherwise.
        """
        y, x = position
        num_samples = 8
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        # Vectorized computation of all sample points
        sample_y = y + self.agent_radius * np.sin(angles)
        sample_x = x + self.agent_radius * np.cos(angles)
        
        # Vectorized array indexing
        grid_y = -np.floor(sample_y).astype(int) - 1
        grid_x = np.floor(sample_x).astype(int)
        
        # Check if any sample point hits a wall
        return not np.any(self.array[grid_y, grid_x] == 1)

    def _are_valid_positions(self, positions: np.ndarray) -> np.ndarray:
        """Vectorized version to check multiple positions at once."""
        num_samples = 8
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        
        # Broadcast positions and angles for vectorized computation
        y_coords = positions[:, 0][:, np.newaxis]  # Shape: (n_positions, 1)
        x_coords = positions[:, 1][:, np.newaxis]  # Shape: (n_positions, 1)
        
        # Compute all sample points for all positions
        sample_y = y_coords + self.agent_radius * np.sin(angles)  # Shape: (n_positions, n_samples)
        sample_x = x_coords + self.agent_radius * np.cos(angles)  # Shape: (n_positions, n_samples)
        
        # Convert to grid coordinates
        grid_y = (-np.floor(sample_y) - 1).astype(int)
        grid_x = np.floor(sample_x).astype(int)
        
        # Check collisions for all positions and samples
        collisions = self.array[grid_y, grid_x] == 1  # Shape: (n_positions, n_samples)
        
        # Return True if position is valid (no collisions), False otherwise
        return ~np.any(collisions, axis=1)  # Shape: (n_positions,)

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
    def __init__(self, array: np.ndarray, step_size=0.4):
        self.maze = Maze(array=array, step_size=step_size)  
        self.start_pos = self.maze.agent_pos                                         # Starting position of the agent
        
        # Pre-compute normalization factors for efficiency
        self.map_size_array = np.array([self.maze.map_height, self.maze.map_width])  # [map_height, map_width] for normalization
        self.map_diagonal_norm = np.linalg.norm(self.map_size_array)                 # Euclidean diagonal of the map (normalization factor)
        self.t = 0

    def step(self, action: int):
        self.maze.step(action)
        self.t += 1
        return self._get_observation(), self.is_done()

    def _get_observation(self):
        """
        Construct the current observation for the agent.
        Includes normalized agent position, distance to goal,
        and obstacle presence in 8 directions.
        Returns:
            np.ndarray: A concatenated feature vector as the agent's state.
        """
        # Basic position information
        agent_pos = self.maze.agent_pos.copy()
        goal_pos = self.maze.goal_pos.copy()
        
        # Normalize positions by map size for better learning
        normalized_agent = agent_pos / self.map_size_array
        
        # Relative goal position (direction to goal)
        goal_distance = np.linalg.norm(goal_pos - agent_pos)

        # Vectorized local obstacle detection (8 directions around agent)
        test_positions = agent_pos + self.maze.action_vectors
        obstacle_info = (~self.maze._are_valid_positions(test_positions)).astype(np.float32)
        
        # Combine all information
        observation = np.concatenate([
            normalized_agent,           # 2 values: normalized agent position  
            [goal_distance / self.map_diagonal_norm], # 1 value: normalized distance to goal
            obstacle_info               # 8 values: obstacle detection in each direction
        ])
        return observation

    def reset(self):
        """
        Reset agent position with noise
        """
        self.maze.agent_pos = self.start_pos + np.random.normal(0, 0.2, size=self.maze.agent_pos.shape)
        self.t = 0
        return self._get_observation()

    def render(self):
        self.maze.render()

    def is_done(self):
        return np.linalg.norm(self.maze.agent_pos - self.maze.goal_pos) < self.maze.goal_radius


class MultiTargetEnvironment(Environment):
    def __init__(self, array: np.ndarray, step_size=0.4):
        super().__init__(array=array, step_size=step_size)
        self.goals = [np.array([15.8, 37]), np.array([17, 18]), np.array([5, 6]), np.array([2, 35])] # List of possible goal positions
        r = np.random.randint(0, len(self.goals)) # Random index to choose current goal and start
        self.goal_pos = self.goals[r]                           # Chosen goal position
        self.maze.agent_pos = self.goals[(r+1)%len(self.goals)] # Starts at different goal position
        self.maze.goal_pos = self.goal_pos # Assign the selected goal position to the maze

    def reset(self):
        """
        Reset the agent to the previous goal, and assign a new random goal.
        Ensures the goal changes every episode for diverse training experiences.
        Returns:
            np.ndarray: The initial observation after environment reset.
        """
        self.t = 0
        old_goal_pos = self.maze.goal_pos  # The previous goal position as new start
        self.maze.agent_pos = old_goal_pos # Set agent's position to old goal
        self.maze.goal_pos = self.goals[np.random.randint(0, len(self.goals))] # Choose a new random goal
        while np.array_equal(old_goal_pos, self.maze.goal_pos): # New random goal must differ from previous
            self.maze.goal_pos = self.goals[np.random.randint(0, len(self.goals))]
        return self._get_observation()
    
    def _get_observation(self):
        """
        Generate the observation vector for the agent.
        Includes normalized agent and goal positions, and their distance.
        Returns:
            np.ndarray: Observation vector.
        """
        # Basic position information
        agent_pos = self.maze.agent_pos.copy()
        goal_pos = self.maze.goal_pos.copy()
        
        # Normalize positions by map size for better learning
        normalized_agent = agent_pos / self.map_size_array
        normalized_goal = goal_pos / self.map_size_array
        
        # Relative goal position (direction to goal)
        goal_distance = np.linalg.norm(goal_pos - agent_pos)

        # Vectorized local obstacle detection (8 directions around agent)
        test_positions = agent_pos + self.maze.action_vectors
        obstacle_info = (~self.maze._are_valid_positions(test_positions)).astype(np.float32)
        
        # Combine all information
        observation = np.concatenate([
            normalized_agent,           # 2 values: normalized agent position
            normalized_goal,            # 2 values: normalized goal position  
            [goal_distance / self.map_diagonal_norm], # 1 value: normalized distance to goal
            # obstacle_info               # 8 values: obstacle detection in each direction
        ])
        return observation


if __name__ == "__main__":
    from time import sleep
    warehouse_map = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
    env = Environment(warehouse_map)
    for _ in range(1):
        env.maze.goal_pos = np.array([16.5, 1.5])
        # env.maze.target_pos = np.array([15.8, 38.5])
        env.render()

        input('next')