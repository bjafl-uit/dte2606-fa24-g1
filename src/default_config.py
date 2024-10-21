"""Default configuration for the application.

Includes parameters for a grid world environment, and default values for the
exploration parameters.

Attributes:
    EXPLORE_PARAMS_DEFAULT (ExploreParamsDefaults): Default exploration
        parameters.
    GRID_WORLD_PREFS (GridWorldPrefs): Default grid world preferences.
"""
from config import ExploreParamsDefaults, GridWorldPrefs

EXPLORE_PARAMS_DEFAULT = ExploreParamsDefaults(
    ALPHA=0.1,
    GAMMA=0.8,
    EPSILON=0.3,
    EPSILON_MIN=0.01,
    EPSILON_MAX=0.9,
    EPSILON_DECAY_RATE=0.001,
    DECAYING_EPSILON=True,
    EP_MAX=15_000,
    STEP_MAX=8_000,
    CONV_RTOL=1e-6,
    CONV_ATOL=1e-8
)

GRID_WORLD_PREFS = GridWorldPrefs(
    REWARDS={
        'g': 0,    # Grass
        'w': -20,  # Water
        'h': -35,  # Hills
        'X': 100   # Goal
    },
    MAP=[
        'whhggw',
        'wwghgg',
        'hgghgh',
        'hggggg',
        'hghghg',
        'Xwwwww'
    ],
    MAP_COLORS={
        'w': (6, 57, 112),
        'h': (110, 15, 15),
        'g': (57, 112, 79),
        'X': (222, 173, 96),
    },
    START_POS=(3, 0),
    REWARD_OUT_OF_BOUNDS=-100,
    MOVES='compass_moves'
)
