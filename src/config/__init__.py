"""Module for configuration and constants.


Attributes:
    ASSETS_PATH (Path): The path to the assets directory.

Classes:
    GridWorldPrefs (NamedTuple): Preference class for initializing
        a new GridWorld.
    ExploreParamsDefaults (NamedTuple): Preference class for default
        exploration parameters.
"""
from pathlib import Path
from .types import GridWorldPrefs, ExploreParamsDefaults

ASSETS_PATH = Path(__file__).parent.parent.parent / 'assets'

__all__ = ['GridWorldPrefs', 'ExploreParamsDefaults', 'ASSETS_PATH']
