from datachallengeg15.dqn import train_dqn, test_dqn_agent
from datachallengeg15.env_dqn import Environment
import numpy as np

# Start training
agent = train_dqn(
    episodes=1000,
    max_steps=300,
    update_freq=10
)

# Load the environment again for testing
maze = np.load("datachallengeg15/warehouse.npy").astype(np.int8)
env = Environment(maze)

# Run the trained agent and visualize
test_dqn_agent(agent, env, episodes=5, max_steps=300)

