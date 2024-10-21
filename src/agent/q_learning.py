"""Q-Learning algorithm implementation.

This module contains the implementation of the Q-Learning algorithm, which is
a model-free reinforcement learning algorithm. The algorithm learns the
optimal policy by updating the Q-values of the state-action pairs according to
the Bellman equation. The Q-values are stored in a Q-table, which is updated
after each transition.
"""
import numpy as np
from typing import Literal, Optional

from environment import MoveSet, Move
from .learning_model import LearningModel, DynamicEpsilon
from .episode import Episode


class QLearning(LearningModel):
    """Q-Learning algorithm implementation.

    Attributes:
        alpha (float): The learning rate.
        gamma (float): The discount factor.
        epsilon (float): The exploration-exploitation trade-off parameter.
        conv_tolerance (tuple[float, float]): The convergence tolerance values.
    """

    def __init__(
            self,
            state_space_shape: Optional[tuple] = None,
            action_set: Optional[MoveSet] = None,
            alpha: float = 0.1,
            gamma: float = 0.8,
            epsilon: float | DynamicEpsilon = 0.5,
            conv_tolerance: tuple[float, float] = (1e-6, 1e-8)
    ) -> None:
        """Initialize the Q-Learning algorithm.

        Args:
            state_space_shape (tuple, optional): The shape of the state space.
            action_set (MoveSet, optional): The set of actions.
            alpha (float, optional): The learning rate.
            gamma (float, optional): The discount factor.
            epsilon (float | DynamicEpsilon, optional): The exploration-
                exploitation trade-off parameter.
            conv_tolerance (tuple[float, float], optional): The convergence
                tolerance values. 
        """
        super().__init__(state_space_shape, action_set)
        self.alpha = alpha
        self.gamma = gamma
        self._epsilon = 0.5           # Init var
        self._dynamic_epsilon = None  # Init var
        self.epsilon = epsilon        # Set epsilon
        self._prev_q_table = None
        self._converged = False
        self._conv_tolerance = conv_tolerance

    @property
    def converged(self) -> bool:
        """Return whether the policy has converged."""
        self._check_initialized()
        if self._converged:
            return True
        if self._prev_q_table is None:
            return False

        if np.allclose(self._q_table, self._prev_q_table,
                       atol=self._conv_tolerance[0],
                       rtol=self._conv_tolerance[1]):
            return True
        return False

    @property
    def epsilon(self) -> float:
        """Return the epsilon value."""
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float | DynamicEpsilon):
        """Set the epsilon value."""
        if isinstance(value, DynamicEpsilon):
            self._epsilon = value()
            self._dynamic_epsilon = value
        else:
            self._epsilon = value
            self._dynamic_epsilon = None

    def decay_epsilon(self, decay_grade: int | Literal['min', 'max']):
        """Decay the epsilon value.

        Args:
            decay_grade (int | str): The grade of decay. If 'min', the epsilon
                value will be set to the minimum value. If 'max', the epsilon
                value will be set to the maximum value. If an integer, the
                epsilon value will be set to the value of the dynamic epsilon
                function at that grade.
        """
        if self._dynamic_epsilon is None:
            raise ValueError("Epsilon is not dynamic.")
        if decay_grade == 'min':
            self._epsilon = self._dynamic_epsilon(-1)
        elif decay_grade == 'max':
            self._epsilon = self._dynamic_epsilon(0)
        else:
            self._epsilon = self._dynamic_epsilon(decay_grade)

    def get_action(self, state: int | tuple) -> Move:
        """Get the best action for the given state.

        Gives the best action according to the epsilon greedy policy. This
        policy gives a 1-epsilon probability to pick a greedy action,
        which will be determined by the largest Q-valye for the next possible
        action. The probability given by the epsilon value, is the probability
        of taking a completely random action.

        Args:
            state (int | tuple): The current state.
        """
        self._check_initialized()
        if np.random.uniform(0, 1) < self._epsilon:
            return super()._get_random_action()
        else:
            state = self._convert_state(state, 'hash')
            action_index = np.argmax(self._q_table[state])
            return self._action_set.get_move_by_index(action_index)

    def on_step_update(
            self,
            state: int | tuple,
            action: Move,
            reward: float,
            next_state: int | tuple
    ) -> None:
        """Learn from the transition.

        Updates the Q-table according to the Bellman equation:
        Q(s, a) = Q(s, a) + α(r + γ * max(Q(s', a')) - Q(s, a)

        Args:
            state (int | tuple): The current state.
            action (Move): The action taken.
            reward (float): The reward received.
            next_state (int | tuple): The next state.
        """
        self._check_initialized()
        alph = self.alpha
        g = self.gamma
        s = self._convert_state(state, 'hash')
        ss = self._convert_state(next_state, 'hash')
        r = reward
        Q = self._q_table
        a = self._action_set.get_index(action.KEY)
        Q[s, a] = Q[s, a] + alph * (r + g * np.max(Q[ss, :]) - Q[s, a])

    def _take_snapshot(self):
        """Take a snapshot of the Q-table."""
        self._prev_q_table = self._q_table.copy()
        self._converged = False

    def on_episode_end(self, episode: Episode):
        """Handle episode end event.

        Takes a snapshot of the Q-table for convergence checking, and decays
        the epsilon value if it is dynamic.

        Args:
            episode (Episode): The episode that ended.
        """
        self._take_snapshot()
        if self._dynamic_epsilon is not None:
            self.decay_epsilon(episode.ep_num + 1)

    def on_start_explore(self):
        """Handle start of exploration event.

        Resets Q-table and the convergence flag.
        """
        self._check_initialized()
        self._converged = False
        self._prev_q_table = None
        self._init_q_table()
