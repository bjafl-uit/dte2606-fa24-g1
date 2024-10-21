"""Define the environment of the grid world.

The environment is defined by the map, the rewards, the start and goal
positions, and the moves the agent can make. The environment is used to
simulate the agent's interaction with the world.
"""
import numpy as np
from typing import Hashable, Union

from .types import Point, SpaceShape, MoveFeedback, Move, CompassMoves
from config import GridWorldPrefs


class GridWorld:
    """Describe the environment of the grid world."""

    def __init__(self, prefs: GridWorldPrefs) -> None:
        """Initialize the environment."""
        self._reward_matrix = np.array(
            [
                [prefs.REWARDS.get(cell, 0) for cell in row]
                for row in prefs.MAP
            ]
        )
        if prefs.GOAL is None:
            self._goal_pos, goal_reward = self._find_goal_data()
        else:
            prefs_goal_pos, goal_reward = prefs.GOAL
            self._goal_pos = Point(*prefs_goal_pos)
        gx, gy = self._goal_pos
        self._reward_matrix[gy, gx] = goal_reward
        self._start_pos = Point(*prefs.START_POS)
        self._reward_oob = prefs.REWARD_OUT_OF_BOUNDS
        if prefs.MOVES == 'compass_moves':
            self._move_set = CompassMoves()
        else:
            self._move_set = prefs.MOVES
        self._shape = SpaceShape(*reversed(self._reward_matrix.shape))
        self._max_xy = Point(*tuple(self._shape)) - Point(1, 1)

    def _find_goal_data(self) -> tuple[Point, float]:
        """Find the position and reward of the goal from the map.

        Uses the max reward in the map to determine the goal position.
        Returns the position and reward of the goal from the reward matrix.
        Will raise an error if there is more than one max value.
        """
        reward = self._reward_matrix.max()
        pos = np.where(self._reward_matrix == reward)
        # Check if there is only one max value.
        if len(pos[0]) > 1:  # Shape is 1D for one max value.
            raise ValueError("Only one goal pos allowed.")
        return Point(*reversed(pos)), reward

    @property
    def start_position(self) -> Point:
        """Return the start position of the robot."""
        return self._start_pos

    @property
    def state_space_shape(self) -> SpaceShape:
        """Return the shape of the map."""
        return self._shape

    @property
    def move_set(self):
        """Return the move set of the grid world."""
        return self._move_set

    def get_reward(self, pos: Point) -> float:
        """Return the reward of the given state."""
        if not self.is_valid_location(pos):
            return self._reward_oob
        return self._reward_matrix[pos.Y, pos.X]

    def make_move(
            self,
            pos: Point,
            move: Union[Move, Hashable]
    ) -> MoveFeedback:
        """Make a move in the grid world."""
        if not isinstance(move, Move):
            move = self._move_set[move]
        new_pos = move(pos)
        reward = self.get_reward(new_pos)
        if not self.is_valid_location(new_pos):
            new_pos = pos
        return MoveFeedback(new_pos, reward, self.is_goal(new_pos))

    def get_random_location(
            self,
            exclude_goal: bool = False
    ) -> Point:
        """Return a random location in the grid world."""
        pos = self._shape.get_random_location()
        while exclude_goal and self.is_goal(pos):
            pos = self._shape.get_random_location()
        return pos

    def is_valid_location(self, pos: Point) -> bool:
        """Check if the location is valid."""
        return pos.check_bounds(self._max_xy)

    def is_goal(self, pos: Point) -> bool:
        """Check if the position is the goal."""
        return pos == self._goal_pos
