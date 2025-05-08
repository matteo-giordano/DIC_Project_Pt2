def reward_fn(grid, agent_cell):
    return 1.0 if agent_cell == grid.target_cell else -0.01
