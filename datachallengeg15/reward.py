def reward_fn(grid, agent_cell):
    return 1.0 if agent_cell == grid.target_cell else -0.01

def reward_dont_revisit(grid, agent_cell):
    return 1.0 if agent_cell == grid.target_cell else -0.01 * (grid.visit_count[agent_cell] + 1)
