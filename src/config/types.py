"""Type definitions for standard prefrences.

Classes:
    GridWorldPrefs: Preference class for initializing
        a new GridWorld
    ExploreParamsDefaults: Set of default values for the exploration parameters
"""
from typing import NamedTuple, Iterable, Optional, Hashable, Union, Literal


class GridWorldPrefs(NamedTuple):
    """Grid world preferences.

    Preference class for the grid world environment. The class is used to
    define the properties of the grid world.

    The MAP is a 2D data structure of states, where each state is given a
    hashable key. This key is used to assign properties to the states. States
    with the same key share the same properties. Each state is uniquely
    identified by its (x, y) coordinates, where (0, 0) is the top-left corner.
    The x- and y-coordinates corresponds to the index of the outer list and
    inner iterable of the MAP respectively. The goal position is given by the
    state with the highest reward, if GOAL_POS is not specified.

    Attributes:
        MAP (list[Iterable[Hashable]]): The map of the grid world.
        START_POS (tuple[int, int]): The (x, y) start position of the agent.
        REWARDS (dict[Hashable, float]): The rewards of the cells.
            The key corresponds to the values in MAP.
        REWARD_OUT_OF_BOUNDS (float): The reward for going out of bounds.
        MAP_COLORS (dict[Hashable, tuple[int, int, int]]):
            The colors of the map. The key corresponds to the values in MAP,
            and colors are given by RGB tuples.
        MOVES (dict[Hashable, tuple[int, int]] | Literal['compass_moves']):
            The moves the agent can make, given by a hashable move key and
            a (dx, dy) move vector. It may also be set to 'compass_moves' to
            select a default NSEW move set.
        GOAL (tuple[Point, float]): The position and reward of the
            goal state. If not specified, the goal position is given by the
            state with the highest reward.

    """

    MAP: list[Iterable[Hashable]]
    START_POS: tuple[int, int]
    REWARDS: dict[Hashable, float]
    REWARD_OUT_OF_BOUNDS: float
    MAP_COLORS: dict[Hashable, tuple[int, int, int]]
    MOVES: Union[dict[Hashable, tuple[int, int]],
                 Literal['compass_moves']] = 'compass_moves'
    GOAL: Optional[tuple[tuple[int, int], float]] = None
    # TRANSITION_MATRIX: None = None  # TODO


class ExploreParamsDefaults(NamedTuple):
    """Set of default values for the exploration parameters.

    Attributes:
        ALPHA (float): The default learning rate.
        GAMMA (float): The default discount factor.
        EPSILON (float): The default exploration rate.
        EPSILON_MIN (float): The default minimum exploration rate.
        EPSILON_MAX (float): The default maximum exploration rate.
        EPSILON_DECAY_RATE (float): The default epsilon decay rate.
        DECAYING_EPSILON (bool): Whether the epsilon should decay.
        EP_MAX (int): The default maximum number of episodes.
        STEP_MAX (int): The default maximum number of steps.
        CONV_RTOL (float): The default relative tolerance for convergence.
        CONV_ATOL (float): The default absolute tolerance for convergence.
    """

    ALPHA: float = 0.1
    GAMMA: float = 0.8
    EPSILON: float = 0.3
    EPSILON_MIN: float = 0.01
    EPSILON_MAX: float = 0.9
    EPSILON_DECAY_RATE: float = 0.001
    DECAYING_EPSILON: bool = True
    EP_MAX: int = 15_000
    STEP_MAX: int = 8_000
    CONV_RTOL: float = 1e-6
    CONV_ATOL: float = 1e-8
