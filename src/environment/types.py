"""Environment data types module.

This module contains the data types used in the grid world environment. The
data types provide named data structures for properties and APIs in the grid
world environment. All the classes are immutable.

Classes:
    Point (NamedTuple): Simple (x, y) point class.
    MoveFeedback (NamedTuple): Defining the feedback data for a move.
    Rgb (NamedTuple): Simple RGB color class.
    SpaceShape (NamedTuple): 2D space shape class (width, height)
    Move (NamedTuple): Move defined by move key and move vector.
    MoveSet: Defining a set of moves by keys and move vectors.
    CompassMoves (NamedTuple, MoveSet): Static move set of cardinal directions.
    GridWorldPrefs: Preference class for initializing a new GridWorld

"""
import random as rnd
from typing import NamedTuple, Iterable, Hashable, Optional, Union, Literal


class Point(NamedTuple):
    """Point class.

    Attributes:
        x (int): The x-coordinate.
        y (int): The y-coordinate.
    """

    X: int
    Y: int

    def __add__(self, other: 'Point') -> 'Point':
        """Add two points together."""
        return Point(self.X + other.X, self.Y + other.Y)
    
    def __sub__(self, other: 'Point') -> 'Point':
        """Subtract two points."""
        return Point(self.X - other.X, self.Y - other.Y)
    
    def check_bounds(
            self,  
            max_point: 'Point', 
            min_point: 'Point' | Literal['origo'] = 'origo') -> bool:
        """Check if the point is within the bounds.
        
        Args:
            max_point (Point): The maximum point of the bounds (inclusive).
            min_point (Point): The minimum point of the bounds (inclusive).
                Defaults to 'origo', which is the point (0, 0).
        """
        if min_point == 'origo':
            min_point = Point(0, 0)
        return (min_point.X <= self.X <= max_point.X 
                and min_point.Y <= self.Y <= max_point.Y)


class MoveFeedback(NamedTuple):
    """Move feedback class.

    Attributes:
        new_pos (Point): The new position of the agent.
        reward (float): The reward of the move.
        goal_reached (bool): Whether the goal was reached.
    """

    NEW_POS: Point
    REWARD: float
    GOAL_REACHED: bool


class Rgb(NamedTuple):
    """RGB class.

    Attributes:
        r (int): The red   value (0-255).
        g (int): The green value (0-255).
        b (int): The blue  value (0-255).
    """

    R: int
    G: int
    B: int


class SpaceShape(NamedTuple):
    """Space shape class.

    Attributes:
        width (int): The width of the space.
        height (int): The height of the space.
    """

    WIDTH: int
    HEIGHT: int

    def get_random_location(self):
        """Return a random location within the space."""
        return Point(
            rnd.randint(0, self.WIDTH - 1),
            rnd.randint(0, self.HEIGHT - 1)
        )


class Move(NamedTuple):
    """Move data type.

    Attributes:
        move_key (Hashable): The key of the move.
        move_vector (Point): The move vector.
    """

    KEY: Hashable
    VECTOR: Point

    def __call__(self, pos: Point) -> Point:
        """Make the move from pos and return new position."""
        return pos + self.VECTOR


class MoveSet(): #TODO: docstring
    """Move set class.

    The move set class defines possible moves for an agent. It is a hashmap of
    move vectors, stored as Point objects, paired with a move key/id. The class
    is immutable.

    Attributes:
        move_keys: Return a list of the move keys. The order of the keys
            is fixed after initialization.
    Methods:
        get_random_move_key: Return a random move key from the set.
        __getitem__: Return the move vector of the key.
        __len__: Return the number of moves in the set.
        __contains__: Check if the key is in the moveset.
        __iter__: Return an iterator of the move keys.
    """

    def __init__(
            self,
            moves: Union[Iterable[Move], dict[Hashable, Point]]
    ) -> None:
        """Initialize the move set, with the moves.

        Args:
            moves (Union[Iterable[Move], dict[Hashable, tuple[int, int]]]):
                The moves of the agent. The moves can be given as an iterable
                of Move objects or a dictionary of hashable keys and move
                vectors (dx, dy).

        """
        if isinstance(moves, dict):
            moves_dict = {}
            for key, value in moves.items():
                moves_dict[key] = Move(key, value)
        else:
            moves_dict = {move.KEY: move for move in moves}
        self._move_keys = list(moves_dict.keys())
        self._moves = dict(moves_dict)
        self._move_index = {key: i for i, key in enumerate(self._move_keys)}

    @property
    def move_keys(self) -> list[Hashable]:
        """Return a list of the move keys."""
        return self._move_keys[:]

    def get_random_move(self) -> Move:
        """Return a random move from the move set."""
        key = rnd.choice(self._move_keys)
        return self._moves[key]

    def get_index(self, key: Hashable) -> int:
        """Return the index of the key in the move set."""
        return self._move_index[key]

    def get_move_by_index(self, index: int) -> Move:
        """Return the move from the index in the move set."""
        key = self._move_keys[index]
        return self._moves[key]

    def __getitem__(self, key: Hashable) -> Point:
        """Return the move vector of the key."""
        return self._moves[key]

    def __len__(self) -> int:
        """Return the number of moves in the set."""
        return len(self._moves)

    def __contains__(self, key: Hashable) -> bool:
        """Check if the key is in the move set."""
        return key in self._moves

    def __iter__(self):
        """Return an iterator of the move keys."""
        return iter(self._move_keys)


class CompassMoves(MoveSet):
    """Compass moves class.

    This is a specialised MoveSet, where the moves are defined by the
    cardinal directions. The class is immutable.

    Attributes:
        NORTH (Point): The north move.
        EAST (Point): The east move.
        SOUTH (Point): The south move.
        WEST (Point): The west move.
        move_keys: Return a list of the move keys. The order of the keys is
            fixed after initialization. Inherited from parent class.

    Methods:
        get_random_move_key: Return a random move key from the set.
            Inherited from parent class.

    Inherited methods:
        __getitem__: Return the move vector of the key.
        __len__: Return the number of moves in the set.
        __contains__: Check if the key is in the moveset.
        __iter__: Return an iterator of the move keys.
    """

    NORTH: Point = Point(0, -1)
    EAST: Point = Point(1, 0)
    SOUTH: Point = Point(0, 1)
    WEST: Point = Point(-1, 0)

    def __init__(self) -> None:
        """Initialize the move set of cardinal directions."""
        moves = {
            "N": CompassMoves.NORTH,
            "E": CompassMoves.EAST,
            "S": CompassMoves.SOUTH,
            "W": CompassMoves.WEST
        }
        super().__init__(moves)

