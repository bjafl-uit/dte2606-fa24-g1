"""Monte Carlo learning model.

The Monte Carlo learning model is a model-free reinforcement learning algorithm
that learns the optimal policy by updating the Q-values of the state-action
pairs after each episode ends. The Q-values are updated by averaging the
returns obtained from the state-action pairs during the episode.
"""
import numpy as np
from typing import Optional,  Iterable, Hashable

from .learning_model import LearningModel
from .episode import Episode


class MonteCarlo(LearningModel):
    """Monte Carlo learning model.

    Attributes:
        gamma (float): The discount factor.
        simulate_policy (bool): Whether to simulate the policy.
    """

    def __init__(
            self,
            state_space_shape: Optional[tuple] = None,
            actions: Optional[Iterable[Hashable]] = None,
            gamma: float = 0.8):
        """Initialize the Monte Carlo learning model.

        Args:
            state_space_shape (tuple, optional): The shape of the state space.
            actions (Iterable[Hashable], optional): The set of actions.
            gamma (float, optional): The discount factor
        """
        super().__init__(state_space_shape, actions)
        self._gamma = gamma
        self._simulate_policy = False
        if self.initialized:
            self._freq_table = self._q_table.copy()

    @property
    def simulate_policy(self) -> bool:
        """Return whether to simulate the policy."""
        return self._simulate_policy

    @simulate_policy.setter
    def simulate_policy(self, value: bool):
        """Set whether to simulate the policy."""
        self._simulate_policy = value

    @property
    def epsilon(self) -> None:
        """Return None - not used in this learning model."""
        return None

    @epsilon.setter
    def epsilon(self, value: float):
        """Enable policy simulation if epsilon is 0."""
        self.simulate_policy = value == 0

    def init_state_action_space(
            self,
            state_space_shape: tuple,
            actions: Iterable[Hashable]
    ) -> None:
        """Initialize the state and action space."""
        super().init_state_action_space(state_space_shape, actions)
        self._freq_table = self._q_table.copy()

    def get_action(self, state: tuple | int) -> str:
        """Return the action for the given state.

        Returns a random action unless policy simulation is enabled.

        Args:
            state (tuple): The current state.
        """
        if self._simulate_policy:
            state = self._convert_state(state, 'hash')
            action_index = np.argmax(self._q_table[state])
            return self._action_set.get_move_by_index(action_index)
        return super()._get_random_action()

    def on_start_explore(self):
        """Reset the q and frequency table."""
        self._init_q_table()
        self._freq_table = self._q_table.copy()

    def on_episode_end(self, episode: Episode):
        """Update the model after each episode."""
        Q = self._q_table
        N = self._freq_table
        G = 0
        t_n = len(episode)
        visited = set()
        for t in reversed(range(t_n)):
            pos, a_key, r = episode[t]
            s = self._convert_state(pos, 'hash')
            a = self._action_set.get_index(a_key)
            if (s, a) not in visited: 
                G = r + G * self._gamma
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / N[s, a]
                visited.add((s, a))

    def on_step_update(
            self,
            state: int | tuple,
            action_key: Hashable,
            reward: float,
            next_state: int | tuple):
        """No action needed for Monte Carlo.

        The Q-table is updated after each episode.
        Method is implemented for consistency with other learning models.

        Returns:
            Callable: A function that does nothing.
        """
        return lambda: None
