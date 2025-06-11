from datachallengeg15.dqn import train_dqn, test_dqn_agent
import yaml

# Load config from YAML
with open("datachallengeg15/config_dqn.yaml", "r") as f:
    dqn_param_dict = yaml.safe_load(f)

# Train the agent (returns both agent and environment)
agent, env = train_dqn(dqn_param_dict)

# Run the trained agent and visualize
test_dqn_agent(agent, env, episodes=5, max_steps=300)