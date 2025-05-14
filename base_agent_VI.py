from .base_agent import BaseAgent
import numpy as np
from world.helpers import action_to_direction
from abc import ABC, abstractmethod

class BaseAgent_VI(BaseAgent):
    def __init__(self, gamma=0.9, theta=1e-4, max_iters=1000):
        """VI agent
            gamma: discount factor. Controls how much the agent values future rewards compared to immediate ones
            theta: convergence threshold. Prevents infinite loops
            max_iters: # of iterations, ensures iteration stops even when theres no convergence
            V: value function (state_index -> value). Each state gets a value estimate. Is initialized to zero and updated during value iteration. It is the expected total future reward the agent can get if it starts in state s and thn follows the optimal policy from there onward 
            policy: (row, col) -> best action (int). Maps each valid position on the grid to the optimal action. Used later in take_action to decide what to do
            state_to_pos: maps each internal state index (int) to its (row, col) grid position. Filled when you scan the grid in run_value_iteration()
            pos_to_state: used to check if a position is valid and finds its corresponding state
        """
        self.gamma = gamma
        self.theta = theta
        self.max_iters = max_iters
        self.V = {}
        self.policy = {}
        self.state_to_pos = {}
        self.pos_to_state = {}
        
    def value_it(self, grid):
        """ 
        
        """
        # state space, use A1_grid.npy
        rows,cols = grid.shape # 15x15 in A1_grid

        #walls
        walls = {1,2} # 1= boundary (56 tiles), 2 = obstacle (48 tiles)

        #target
        target = 3 # (1 tile)

        # Map valid positions to state indices
        idx = 0
        for i in range(rows):
            for j in range(cols):
                # all valid positions (non-walls) get an entry;
                if grid[i,j] not in walls:
                    self.state_to_pos[idx] = (i,j) # stateID: (grid pos), i.e. {0: (1,1)} 
                    self.pos_to_state[(i,j)] = idx # (grid pos), stateID, i.e. {(1,1): 0}
                    self.V[idx] = 0 # stateID: value 
                    idx += 1
        
        # Value iteration; do Bellman update to define value of a state
        for _ in range(self.max_iters):
            delta = 0 # checks how much values change (for convergence)
            new_V = self.V.copy() # holds updates value for this iteration

            # Iterate over all valid states
            # for each valid state s, get its (i,j) grid position
            for s in self.V:
                i, j = self.state_to_pos[s]
                # use max_ al to track best value achievable from this state
                max_val = float('-inf')

                # try 4 possible actions (0=down, 1=up, 2=left, 3=right)
                for a in range(4):
                    di, dj = action_to_direction(a)
                    # calculate next position (ni, nj) after moving
                    ni, nj = i + di, j + dj

                    # If the move is valid (so if the move is in pos_to_state)
                    if (ni, nj) in self.pos_to_state:
                        # s' = next state index 
                        s_prime = self.pos_to_state[(ni, nj)]
                        # check what the reward is for the move 
                        reward = self.reward_function(grid, (ni, nj))
                        # val = how good the move is
                        val = reward + self.gamma * self.V[s_prime]
                    # If the move is invalid (off-grid or wall), you still get a penalty but no future value because you can't move    
                    else:
                        val = self.reward_function(grid, (ni, nj))
                    # Keep the best alue    
                    max_val = max(max_val, val)
                # update value     
                new_V[s] = max_val
                # track how much value changed compared to before delta
                delta = max(delta, abs(self.V[s] - max_val))
            # update V
            self.V = new_V
            # if no states value changed, the agent is not learning anything new anymore and thus found the optimal values 
            if delta < self.theta:
                break

        # Policy
        # After value function has converged, you want to extract the optimal policy
        # i.e., for every position (i,j) what is the best action to take?
        # loop over all valid state indices 
        # i.e. find best action from each thate
        for s in self.V:
            i, j = self.state_to_pos[s]
            best_action = None
            best_value = float('-inf')
            for a in range(4):
                di, dj = action_to_direction(a)
                ni, nj = i + di, j + dj
                if (ni, nj) in self.pos_to_state:
                    s_prime = self.pos_to_state[(ni, nj)]
                    reward = self.reward_function(grid, (ni, nj))
                    val = reward + self.gamma * self.V[s_prime]
                else:
                    val = self.reward_function(grid, (ni, nj))
                if val > best_value:
                    best_value = val
                    best_action = a
            self.policy[(i, j)] = best_action

    #@abstractmethod
    def take_action(self, state: tuple[int, int]) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """

        return self.policy.get(state, 0)
        #raise NotImplementedError
    
    #@abstractmethod
    def update(self, state: tuple[int, int], reward: float, action: int):
        """Any code that processes a reward given the state and updates the agent.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        pass
        #raise NotImplementedError

    def reward_function(self, grid, pos):
        """"
        define reward function urself
        """
        i, j = pos
        rows, cols = grid.shape
        if 0 <= i < rows and 0 <= j < cols:
            tile = grid[i, j]
            if tile == 0: return 1  # Empty
            if tile in {1, 2}: return -1  # Obstacle or boundary
            if tile == 3: return 10  # Target
        return -5  # Invalid (off-grid or unknown)    