"""This module implements the environment.

The module contains the GridWorld class, a basic grid-based environment for a
agent. The class defines the available states and actions for the agent, the
state transitions and rewards, including a definition of the goal state. The
submodule types contains a set of immutable data classes for the environment,
that simplifies the GridWorld's API. The properties of a GridWorld object is
set by an instance of the GridWorldPrefs class.

Classes:
    GridWorld: The grid world environment.
    Point: A simple (x, y) point class.
    SpaceShape: A 2D space shape class (width, height).
    MoveFeedback: The feedback data for a move.
    MoveSet: A set of moves by keys and vectors.
    CompassMoves: A static move set of cardinal directions.
    Move: A move defined by move key and move vector.
    Rgb: A simple RGB color class.
"""
from .grid_world import GridWorld
from .types import Point, SpaceShape, MoveFeedback, MoveSet, CompassMoves, \
    Move, Rgb

__all__ = ['GridWorld', 'Point', 'SpaceShape', 'MoveFeedback', 'Move',
           'MoveSet', 'CompassMoves', 'Rgb']
