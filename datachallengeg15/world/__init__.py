from pathlib import Path

from datachallengeg15.world.grid import Grid
from datachallengeg15.world.gui import GUI
from datachallengeg15.world.environment import Environment


GRID_CONFIGS_FP = Path(__file__).parents[1].resolve() / Path("grid_configs")
GRID_CONFIGS_FP.mkdir(parents=True, exist_ok=True)

__all__ = ["GRID_CONFIGS_FP", "Grid", "GUI", "Environment"]

