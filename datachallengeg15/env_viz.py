import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.colors as mcolors
from agent import BaseAgent
from grid import Grid
import numpy as np


def visualize_q_values(agent: BaseAgent, grid: Grid, start: tuple[int, int], goal: tuple[int, int]):
    maze_binary = grid.array.T
    directions = {(-1, 0): 'L', (1, 0): 'R', (0, -1): 'U', (0, 1): 'D'}
    q_map = {}
    for (s, a), q in agent.q_table.items():
        dx, dy = a[0] - s[0], a[1] - s[1]
        dir = directions.get((dx, dy))
        if dir:
            if s not in q_map:
                q_map[s] = {}
            q_map[s][dir] = q

    all_q = [q for v in q_map.values() for q in v.values()]
    norm = mcolors.Normalize(vmin=min(all_q), vmax=max(all_q))
    cmap = plt.cm.viridis

    H, W = maze_binary.shape
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(maze_binary, cmap='gray_r', origin='upper', extent=(0, W, H, 0))

    for (x, y), qs in q_map.items():
        if maze_binary[y, x] != 0:
            continue

        cx, cy = x, y
        if 'U' in qs:
            color = cmap(norm(qs['U']))
            tri = [(cx, cy), (cx + 1, cy), (cx + 0.5, cy + 0.5)]
            ax.add_patch(patches.Polygon(tri, color=color))
        if 'D' in qs:
            color = cmap(norm(qs['D']))
            tri = [(cx, cy + 1), (cx + 1, cy + 1), (cx + 0.5, cy + 0.5)]
            ax.add_patch(patches.Polygon(tri, color=color))
        if 'L' in qs:
            color = cmap(norm(qs['L']))
            tri = [(cx, cy), (cx, cy + 1), (cx + 0.5, cy + 0.5)]
            ax.add_patch(patches.Polygon(tri, color=color))
        if 'R' in qs:
            color = cmap(norm(qs['R']))
            tri = [(cx + 1, cy), (cx + 1, cy + 1), (cx + 0.5, cy + 0.5)]
            ax.add_patch(patches.Polygon(tri, color=color))

    # Highlight start and goal squares
    sx, sy = start
    gx, gy = goal
    ax.add_patch(patches.Rectangle((sx, sy), 1, 1, edgecolor='red', facecolor='red', alpha=0.5))
    ax.add_patch(patches.Rectangle((gx, gy), 1, 1, edgecolor='red', facecolor='red', alpha=0.5))

    # Centered path
    path = agent.extract_policy_path(start, goal)
    if path:
        px, py = zip(*[(x + 0.5, y + 0.5) for (x, y) in path])
        ax.plot(px, py, color='red', linewidth=2, marker='o', markersize=3)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect('equal')

    for x in range(W + 1):
        ax.axvline(x, color='lightgray', linewidth=0.5)
    for y in range(H + 1):
        ax.axhline(y, color='lightgray', linewidth=0.5)

    ax.set_title("Q-values Heatmap & Optimal Path")
    ax.axis('off')
    plt.tight_layout()
    plt.show()

