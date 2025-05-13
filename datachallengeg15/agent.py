from abc import ABC, abstractmethod

import numpy as np 
import networkx as nx
import random
from collections import defaultdict


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
        self.V = {}       # value function: state -> value
        self.policy = {}  # mapping: state -> best next state
        # Actions: 0=Down, 1=Up, 2=Left, 3=Right (matches grid.py move_agent comment)
        self.actions = [
            (0, 1),    # Down
            (0, -1),   # Up
            (-1, 0),   # Left
            (1, 0),    # Right
        ]

    def solve(self, stochasticity=0.0):
        """
        Perform Value Iteration to compute the optimal value function and policy.

        Args:
            stochasticity (float): The probability that the intended action fails and
            the agent executes a random other action instead. For example, if 
            stochasticity=0.1, then the intended action is followed with 0.9 probability,
            and the remaining 0.1 is distributed equally among the other 3 directions.
        """
        # Initialize the value function for all states to 0
        self.V = {s: 0.0 for s in self.graph.nodes}

        while True:
            delta = 0.0  # Tracks the largest value change in this iteration

            # Iterate over all states in the graph (i.e., reachable positions)
            for state in self.graph.nodes:
                best_value = float("-inf")  # Initialize best value for this state
                best_action = None          # Store the corresponding best action

                # Loop over all intended movement directions (up, down, left, right)
                for intended_a in self.actions:
                    v = 0.0  # Accumulate expected value for this action

                    # Simulate stochastic effects: loop over all actual actions taken (4 directions)
                    for actual_a in self.actions:
                        # Determine the probability of taking actual_a given intended_a
                        # For example, if stochasticity=0.1, then the intended action is followed with 0.9 probability,
                        # and the remaining 0.1 is distributed equally among the other 3 directions.
                        if actual_a == intended_a:
                            prob = 1 - stochasticity
                        else:
                            prob = stochasticity / (len(self.actions) - 1)

                        # Compute next state after taking action actual_a
                        next_state = (state[0] + actual_a[0], state[1] + actual_a[1])
                        # If the resulting state is invalid (e.g., wall), stay in place
                        if next_state not in self.graph:
                            next_state = state

                        # Get the reward and update expected v-value
                        reward = self.reward_fn(self.grid, next_state)
                        v += prob * (reward + self.gamma * self.V[next_state])

                    # Track the best action and value across all intended directions
                    if v > best_value:
                        best_value = v
                        # The best action is the intended direction that gives max V
                        best_action = (state[0] + intended_a[0], state[1] + intended_a[1])

                # If the best action leads to an invalid state, stay in place
                if best_action not in self.graph:
                    best_action = state

                # Update the value function and policy
                delta = max(delta, abs(self.V[state] - best_value))
                self.V[state] = best_value
                self.policy[state] = best_action

            # If all updates are below the convergence threshold, stop
            if delta < self.theta:
                break
