from abc import ABC, abstractmethod

import numpy as np 
import networkx as nx
import random
from collections import defaultdict
from tqdm import trange

class BaseAgent(ABC):
    def __init__(self):
        """Base agent. All other agents should build on this class.

        As a reminder, you are free to add more methods/functions to this class
        if your agent requires it.
        """

    @abstractmethod
    def take_action(self, state: tuple[int, int]) -> int:
        """Any code that does the action should be included here.

        Args:
            state: The updated position of the agent.
        """
        raise NotImplementedError
    
    @abstractmethod
    def update(self, state: tuple[int, int], reward: float, action: int):
        """Any code that processes a reward given the state and updates the agent.

        Args:
            state: The updated position of the agent.
            reward: The value which is returned by the environment as a
                reward.
            action: The action which was taken by the agent.
        """
        raise NotImplementedError
  

class TabularQLearningAgent(BaseAgent):
    def __init__(self, graph: nx.Graph, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.graph = graph
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}
        
    def get_action(self, state):
        return list(self.graph.neighbors(state))
    
    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.)
    
    def take_action(self, state):
        actions = self.get_action(state)
        if not actions:
            return None
        if np.random.rand() < self.epsilon:
            return random.choice(actions)
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)
    
    def update(self, state, action, reward, next_state):
        next_actions = self.get_action(next_state)
        max_next_q = max([self.get_q(next_state, a) for a in next_actions], default=0.)
        current_q = self.get_q(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[(state, action)] = new_q
        
    def extract_policy_path(self, start, end, max_steps=1000):
        state = start
        path = [state]
        visited = set()
        for _ in range(max_steps):
            if state == end:
                break
            actions = self.get_action(state)
            if not actions:
                break
            q_values = [self.get_q(state, a) for a in actions]
            best_action = actions[np.argmax(q_values)]
            if best_action in visited:
                break
            visited.add(best_action)
            state = best_action
            path.append(state)
        return path
        
        

class MonteCarloAgent(BaseAgent):
    def __init__(self, graph: nx.Graph, epsilon: float = 0.1, gamma: float = 0.99, epsilon_decay: float = 0.995, epsilon_min: float = 0.01):
        """
        Monte Carlo Control Agent using on-policy first-visit method with epsilon-soft policy.
        """
        self.graph = graph
        self.epsilon = epsilon
        self.initial_epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        # Q-value table: Q[state][action] = value
        self.q_table = {}

        # For tracking returns for averaging
        self.returns = defaultdict(list)  # returns[(state, action)]

        # To store episode trajectory
        self.episode = []

    def get_action(self, state):
        """Get available actions from the current state."""
        return list(self.graph.neighbors(state))
    
    def get_q(self, state, action):
        """Get Q-value for a state-action pair."""
        return self.q_table.get((state, action), 0.0)

    def take_action(self, state):
        """Choose an action according to the current epsilon-greedy policy."""
        actions = self.get_action(state)
        if not actions:
            return None
            
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return random.choice(actions)
        
        # Greedy action selection
        q_values = [self.get_q(state, a) for a in actions]
        max_q = max(q_values)
        best_actions = [a for a, q in zip(actions, q_values) if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state):
        """Append to the episode buffer. Called every step."""
        self.episode.append((state, action, reward))

    def end_episode(self):
        """Called after the episode ends. Perform Monte Carlo updates."""
        G = 0
        visited = set()

        for t in reversed(range(len(self.episode))):
            state, action, reward = self.episode[t]
            G = self.gamma * G + reward

            if (state, action) not in visited:
                visited.add((state, action))
                self.returns[(state, action)].append(G)
                self.q_table[(state, action)] = np.mean(self.returns[(state, action)])

        # Clear the episode buffer
        self.episode.clear()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def extract_policy_path(self, start, end, max_steps=1000):
        """Extract the best policy path from start to end."""
        state = start
        path = [state]
        visited = set()
        for _ in range(max_steps):
            if state == end:
                break
            actions = self.get_action(state)
            if not actions:
                break
            q_values = [self.get_q(state, a) for a in actions]
            best_action = actions[np.argmax(q_values)]
            if best_action in visited:
                break
            visited.add(best_action)
            state = best_action
            path.append(state)
        return path

class ValueIterationAgent:
    def __init__(self, reward_fn, graph: nx.Graph, gamma: float = 0.99, theta: float = 1e-4, ):
        """
        Value Iteration Agent with minimal setup, matching Q-learning structure.

        Args:
            graph: networkx Graph object
            gamma: discount factor
            theta: convergence threshold
        """
        self.graph = graph
        self.gamma = gamma
        self.theta = theta
        self.reward_fn = reward_fn
        self.iterations = 0  # optional, used in plan_on_map
        self.V = {}       # value function: state -> value
        self.policy = {}  # mapping: state -> best next state
        # Actions: 0=Down, 1=Up, 2=Left, 3=Right (matches grid.py move_agent comment)
        self.actions = [
            (0, 1),    # Down
            (0, -1),   # Up
            (-1, 0),   # Left
            (1, 0),    # Right
        ]

    def take_action(self, state: tuple[int, int]) -> tuple[int, int]:
        """
        Deterministically choose the best action from the learned policy.
        Used during evaluation (rollout).
        """
        return self.policy.get(state, state)

    def solve(self, grid, sigma=0.0, max_iterations=5000):
        """
        Perform Value Iteration to compute the optimal value function and policy.

        Args:
            -grid: Grid object (for reward function access).
            -sigma (float): The probability that the intended action fails and
                the agent executes a random other action instead. For example, if 
                sigma=0.1, then the intended action is followed with 0.9 probability,
                and the remaining 0.1 is distributed equally among the other 3 directions.
            -max_iterations (int): maximum number of iterations allowed.
        """
        # Initialize the value function for all states to 0
        self.V = {s: 0.0 for s in self.graph.nodes}
        self.iterations = 0  # Initialize iteration counter

        # pbar = trange(max_iterations, desc="Value Iteration", leave=True)
        #for _ in pbar:
        for i in range(max_iterations):
            delta = 0.0  # Tracks the largest value change in this iteration
            new_V = self.V.copy() # Copy the 
            # Iterate over all states in the graph (i.e., reachable positions)
            for state in self.graph.nodes:
                # Only consider legal intended actions (neighbors)
                legal_next_states = list(self.graph.neighbors(state))
                if not legal_next_states:
                    continue
                
                best_value = float("-inf")  # Initialize best value for this state
                best_action = None          # Store the corresponding best action

                # Loop over all intended movement directions (up, down, left, right)
                for intended in legal_next_states:
                    v = 0.0  # Accumulate expected value for this action
                    # Simulate stochastic effects: loop over all actual actions taken
                    for actual in legal_next_states:
                        # Determine the probability of taking actual_a given neighbor
                        # For example, if sigma=0.1, then the intended action is followed with 0.9 probability,
                        # and the remaining 0.1 is distributed equally among the other 3 directions.
                        if actual == intended:
                            prob = 1 - sigma
                        else:
                            prob = sigma / (len(legal_next_states) - 1)

                        # update expected v-value
                        # v = SUM(P(s_next | s, a) * (reward(s, a, s_next) + gamma * V[s_next]))
                        reward = self.reward_fn(grid, actual)
                        v += prob * (reward + self.gamma * self.V[actual])

                    # best action and value among all intended directions
                    # V[s] = MAX(v)
                    if v > best_value:
                        best_value = v
                        # The best action is the intended direction that gives max V[s]
                        best_action = intended

                # If the best action leads to an invalid state, stay in place
                if best_action is None or best_action not in self.graph:
                    best_action = state

                # Update the value function and policy
                new_V[state] = best_value
                self.policy[state] = best_action
                delta = max(delta, abs(self.V[state] - best_value))

            self.V = new_V
            self.iterations += 1
            #pbar.set_postfix(delta=delta)
            # If all updates are below the convergence threshold, stop
            if delta < self.theta:
                break

    def extract_policy_path(self, start: tuple[int, int], goal: tuple[int, int], max_steps: int = 1000) -> list[tuple[int, int]]:
        """
        Extract the greedy path from start to goal by following the learned policy.
        This method mimics the interface and behavior of Q-learning agent's extract_policy_path().
        It is used to evaluate the quality of the learned policy by checking whether it can
        successfully guide the agent from the starting cell to the goal cell.

        Args:
            start (tuple): The starting cell (x, y).
            goal (tuple): The target/goal cell (x, y).
            max_steps (int): Maximum number of steps to allow, in case of loops or dead-ends.

        Returns:
            path (List[tuple]): List of states visited from start to goal (inclusive).
        """

        state = start  # Initialize current state
        path = [state]  # Initialize path with starting state
        visited = set()  # Keep track of visited states to detect loops

        for _ in range(max_steps):
            # If the goal is reached, return the path
            if state == goal:
                break

            # If the current state is not in the learned policy, stop (no decision available)
            if state not in self.policy:
                break

            # Get the next state from the learned deterministic policy
            next_state = self.policy[state]

            # If we are stuck in a loop or invalid transition, stop
            if next_state in visited:
                break

            visited.add(next_state)
            path.append(next_state)
            state = next_state

        return path
    
    def get_value_function(self) -> dict:
        """
        Return the current value function as a dictionary.

        This is equivalent to get_q_table() in Q-learning agents and is useful
        for visualization, evaluation, and analysis of the learned value function.

        Returns:
            dict: A mapping from state (tuple) to its estimated value V(s).
        """
        return self.V
