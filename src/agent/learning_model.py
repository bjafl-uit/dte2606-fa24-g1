"""."""
import numpy as np
from typing import Literal, Optional
from dataclasses import dataclass

from .episode import Episode
from environment import MoveSet, Move


@dataclass
class DynamicEpsilon:
    """Dynamic epsilon class for handling epsilon decay.

    Attributes:
        epsilon_min (float): The minimum epsilon value.
        epsilon_max (float): The maximum epsilon value.
        epsilon_decay_rate (float): The epsilon decay rate
    """

    epsilon_min: float = 0.01
    epsilon_max: float = 1.0
    epsilon_decay_rate: float = 0.0001

    def __call__(self, decay_grade: int = 0) -> float:
        """Return the epsilon value for the given episode.

        If decay_grade is 0, the epsilon_max value is returned.
        If decay_grade is -1, the epsilon_min value is returned.
        Otherwise, the epsilon value is determined by an exponential decay
        function. The decay grad should be a positive integer. It may also
        be -1 or 0, in which case the epsilon_min or epsilon_max value is
        returned, respectively.

        Args:
            decay_grade (int): The grade of the decay. Defaults to 0.
                Using episode number as the decay grade is recommended.

        Returns:
            float: The epsilon value for the given argument.

        Raises:
            ValueError: If the decay grade is less than -1.
        """
        if decay_grade < -1:
            raise ValueError("Invalid decay grade.")
        if decay_grade == 0:
            return self.epsilon_max
        elif decay_grade == -1:
            return self.epsilon_min
        return (
            self.epsilon_min + (self.epsilon_max - self.epsilon_min)
            * np.exp(-self.epsilon_decay_rate * decay_grade)
        )


class LearningModel:

    def __init__(
            self,
            state_space_shape: Optional[tuple] = None,
            action_set: Optional[MoveSet] = None
    ) -> None:
        self._action_set = action_set
        self._state_space_shape = state_space_shape

        self._n_states: Optional[int] = None
        self._q_table: Optional[np.ndarray] = None

        if self.initialized:
            self._init_state_action_space()

    def init_state_action_space(
            self,
            state_space_shape: tuple,
            action_set: MoveSet
    ) -> None:
        """Initialize the state and action space.

        Args:
            state_space_shape (tuple): The shape of the state space.
            action_set (MoveSet): The action set.
        """
        if self._state_space_shape is not None or self._action_set is not None:
            raise ValueError("State and action space already initialized.")
        self._state_space_shape = state_space_shape
        self._action_set = action_set
        self._init_state_action_space()

    def _init_state_action_space(self):
        self._n_states = int(np.prod(self._state_space_shape, axis=0))
        self._init_q_table()

    @property
    def initialized(self) -> bool:
        """Return whether the model is initialized."""
        return (self._action_set is not None
                and self._state_space_shape is not None)

    @property
    def converged(self) -> bool:
        """Return whether the model has converged.

        This method should be implemented by the subclass. This property
        returns False by default if not overridden.
        """
        return False

    @property
    def epsilon(self) -> float | None:
        """Return the epsilon value.

        This method should be implemented by the subclass. This property
        returns None by default if not overridden.
        """
        return None

    def get_action(self, state: int | tuple) -> Move:
        """Get the action for the given state.

        This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def _get_random_action(self):
        return self._action_set.get_random_move()

    def on_step_update(
            self,
            state: int | tuple,
            action: Move,
            reward: float,
            next_state: int | tuple):
        """Update the model after each step.

        This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def on_start_explore(self):
        """Reset the model before exploration.

        This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def on_episode_end(self, episode: Episode):
        """Update the model after each episode.

        This method should be implemented by the subclass.
        """
        raise NotImplementedError

    def hash_state(self, state: tuple) -> int:
        """Hash the state tuple.

        The tuple of states are hashed according to the number of possible
        states in the environment, given by the state_space_shape property.
        The hashed state corresponds to the state's row index in the Q-table.

        Args:
            state (tuple): The state to hash.

        Returns:
            int: The hashed state.

        Raises:
            ValueError: If the state shape doesn't match the policy's state
                space shape, or if a value in state is out of bounds.
        """
        self._check_initialized()
        hashed = 0
        if len(state) != len(self._state_space_shape):
            raise ValueError("Invalid state shape.")
        for i, (s, n) in enumerate(zip(state, self._state_space_shape)):
            if 0 > s >= n:
                raise ValueError(f"The state[{i}] is out of bounds.")
            hashed = hashed * n + s
        return hashed

    def unhash_state(self, hashed_state: int) -> tuple:
        """Unhash the position.

        Un-hash the hashed state to the original state tuple, given the state
        space shape of the policy. The hashed state must be within the bounds
        of the state space shape.

        Args:
            state (int): The hashed state.

        Returns:
            tuple: The unhashed state.

        Raises:
            ValueError: If the hashed state is out of bounds.
        """
        self._check_initialized()
        if 0 > hashed_state >= len(self._state_space_shape):
            raise ValueError("The hashed state is out of bounds.")
        state = []
        for n in reversed(self._state_space_shape):
            state.append(hashed_state % n)
            hashed_state //= n
        return tuple(reversed(state))

    def _convert_state(
            self,
            state: tuple | int,
            action: Literal['hash', 'unhash'] = 'hash'
    ) -> int | tuple:
        """Convert the state to the appropriate format if necessary."""
        if action == 'hash':
            if isinstance(state, tuple):
                return self.hash_state(state)
            elif isinstance(state, int):
                return state
        elif action == 'unhash':
            if isinstance(state, int):
                return self.unhash_state(state)
            elif isinstance(state, tuple):
                return state
        raise ValueError("Invalid state type or action.")

    def get_unhashed_q_table(self):
        """Return the unhashed Q-table, with state tuples as keys."""
        self._check_initialized()
        width, height = self._state_space_shape
        q_dict = {}
        for x in range(width):
            for y in range(height):
                q_dict[(x, y)] = {}
                for i, a in enumerate(self._action_set.move_keys):
                    q_dict[(x, y)][a] = self._q_table[
                        self.hash_state((x, y)), i]
        return q_dict

    def get_best_actions_table(self):
        """Return the best actions table."""
        width, height = self._state_space_shape
        table = [['' for _ in range(width)] for _ in range(height)]
        for y in range(height):
            for x in range(width):
                state_index = self._convert_state((x, y), 'hash')
                action_index = self._q_table[state_index].argmax()
                table[y][x] = self._action_set.get_move_by_index(
                    action_index).KEY
        return table

    def _check_initialized(self) -> None:
        """Check if the model is properly initialized."""
        if not self.initialized:
            raise ValueError("Learning model not properly initialized.")

    def _init_q_table(self):
        """Initialize the Q-table."""
        self._q_table = np.zeros((self._n_states, len(self._action_set)))
