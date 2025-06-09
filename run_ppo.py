from datachallengeg15.ppo import test_trained_agent

test_trained_agent(
    model_path="ppo_maze_model.pth",
    episodes=2,
    max_steps=250
)
