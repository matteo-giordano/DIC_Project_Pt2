import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as patches
from shapely.geometry import Polygon
from agent import BaseAgent
from grid import ContinuousWorld as Grid

def visualize_q_values(agent: BaseAgent, grid: Grid, start: tuple[float, float], goal: tuple[float, float]):
    # Prepare color normalization
    if hasattr(agent, "q_table"):
        q_map = {}
        for (s, a), q in agent.q_table.items():
            q_map[s] = max(q_map.get(s, float('-inf')), q)
        values = list(q_map.values())
        title = "Q-values Heatmap & Policy Path"
    elif hasattr(agent, "get_value_function"):
        q_map = agent.get_value_function()
        values = list(q_map.values())
        title = "Value Function Heatmap & Policy Path"
    else:
        print("Agent has no Q-values or value function.")
        return

    if not values:
        print("No values to visualize.")
        return

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(values), vmax=max(values))

    # Start plotting
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw obstacles
    for poly in grid.obstacles:
        x, y = poly.exterior.xy
        ax.fill(x, y, color='black')

    # Plot Q/Value points
    for (x, y), val in q_map.items():
        ax.plot(x, y, 'o', color=cmap(norm(val)), markersize=5)

    # Plot path from agent
    if hasattr(agent, "extract_policy_path"):
        path = agent.extract_policy_path(start, goal)
        if path:
            px, py = zip(*path)
            ax.plot(px, py, color='red', linewidth=2, marker='o', markersize=3, label="Policy Path")

    # Draw start and goal markers
    ax.plot(*start, 'go', markersize=10, label='Start')
    ax.plot(*goal, 'ro', markersize=10, label='Goal')

    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect('equal')
    ax.set_title(title)
    ax.axis('off')
    ax.legend()
    plt.tight_layout()
    plt.show()