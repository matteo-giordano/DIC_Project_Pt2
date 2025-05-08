import numpy as np
import networkx as nx

class Grid:
    def __init__(self, array: np.ndarray, start_cell: tuple[int, int]):
        if not self.validate_array(array):
            raise ValueError("Invalid array")
        self.array = array
        self.graph = self.build_graph()
        self.n_cols = array.shape[0]
        self.n_rows = array.shape[1]
        self.target_cell = tuple(np.argwhere(array == 3)[0][::-1])
        self.start_cell = start_cell
        self.agent_cell = start_cell
        self.graph = self.build_graph()

    def validate_array(self, array: np.ndarray):
        assert array.ndim == 2, "Array must be 2D"
        assert array.shape[0] > 0 and array.shape[1] > 0, "Array must have non-zero dimensions"
        assert array.dtype == np.int8, "Array must be of type int8"
        assert array.min() >= 0 and array.max() <= 3, "Array values must be between 0 and 3"
        assert array.shape[0] % 2 == 1 and array.shape[1] % 2 == 1, "Array dimensions must be odd"
        assert np.all(array[0, :] == 1) and np.all(array[-1, :] == 1), "Array must have boundaries"
        assert np.all(array[:, 0] == 1) and np.all(array[:, -1] == 1), "Array must have boundaries"
        return True
    
    def move_agent(self, action: tuple[int, int]):
        # 0: Move down, 1: Move up, 2: Move left, 3: Move right
        assert self.array[action] != 1, "Invalid action"
        self.agent_cell = action

    def is_done(self):
        return self.agent_cell == self.target_cell
    
    def reset(self):
        self.agent_cell = self.start_cell
        return self.agent_cell

    def build_graph(self):
        G = nx.Graph()
        h, w = self.array.shape
        for y in range(h):
            for x in range(w):
                if self.array[y, x] == 0 or self.array[y, x] == 3:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx_, ny_ = x + dx, y + dy
                        if 0 <= nx_ < w and 0 <= ny_ < h and self.array[ny_, nx_] == 0:
                            G.add_edge((y, x), (ny_, nx_))
        return G

        