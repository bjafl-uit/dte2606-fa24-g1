"""Contains the Episode class."""


class Episode:
    """Episode class for storing the history of an episode.

    An episode is a sequence of states, actions, and rewards that the robot
    experiences while exploring the environment. Data may be added or returned
    in pairs of these three values. The last step in the episode only has a
    location, and not an action or reward. The last position is stored in a
    separate property to keep the other lists in sync.

    Any episodes may be recorded with different epsilon values and the agent
    may reach the goal during an episode. The goal position is stored if the
    goal is reached. The epsilon value, and episode number are also stored.

    Methods:
        add_move: Add a move to the episode.
        __iter__: Iterate over states in the episode.
        __str__: Return a string representation of the episode.

    Properties:
        num_moves: Return the number of moves made this episode.
        total_reward: Return the total cumulative reward this episode.
        positions: Return the positions traversed this episode.
        moves: Return the moves made in the episode.
        rewards: Return the rewards for each action state pair.
        ep_num: Return the episode number.
        epsilon: Return the epsilon value used for this episode.
    """

    def __init__(
            self,
            ep_num: int,
            epsilon: float | str,
    ):
        """Initialize the episode.

        Args:
            ep_num (int): The episode number.
            epsilon (float): The epsilon value.
        """
        self._ep_num = ep_num
        self._epsilon = epsilon
        self._positions: list[tuple[int, int]] = []
        self._moves: list[str] = []
        self._rewards: list[float] = []
        # In the last state the robot won't move, so we need to store the last
        # position separately to keep moves, rewards, and positions in sync.
        self._last_pos: tuple[int, int] | None = None

    @property
    def num_moves(self) -> int:
        """Return the number of moves made this episode.

        Returns:
            int: The number of moves.
        """
        return len(self.moves)

    @property
    def total_reward(self) -> float:
        """Return the total cumulative reward this episode.

        Returns:
            float: The total reward.
        """
        return sum(self.rewards)

    @property
    def positions(self) -> list[tuple[int, int]]:
        """Return the positions traversed this episode.

        Returns:
            list[tuple[int, int]]: The positions traversed.
        """
        if self._last_pos is not None:
            return self._positions + [self.last_pos]
        else:
            return self._positions
        
    @property
    def last_pos(self) -> tuple[int, int]:
        """Return the last position in the episode.

        Returns:
            tuple[int, int]: The last position.
        """
        return self._last_pos
    
    @last_pos.setter
    def last_pos(self, pos: tuple[int, int]) -> None:
        """Set the last position in the episode.

        Args:
            pos (tuple[int, int]): The last position.
        """
        self._last_pos = pos

    @property
    def moves(self) -> list[str]:
        """Return the moves made in the episode.

        Returns:
            list[str]: The moves made in the episode.
        """
        return self._moves

    @property
    def rewards(self) -> list[float]:
        """Return the rewards for each action state pair.

        Returns:
            list[float]: The rewards for each action state pair.
        """
        return self._rewards

    @property
    def ep_num(self) -> int:
        """Return the episode number.

        Returns:
            int: The episode number
        """
        return self._ep_num

    @property
    def epsilon(self) -> float:
        """Return the epsilon value used for this episode.

        Returns:
            float: The epsilon value.
        """
        return self._epsilon

    def __iter__(self):
        """Iterate over states in the episode.

        Provides an iterator with the tuple of position, move, and reward for
        all steps in the episode.

        Returns:
            Iterator: An iterator over the states in the episode.
        """
        return zip(self.positions, self.moves, self.rewards)
    
    def __len__(self) -> int:
        """Return the number of steps in the episode.
        
        The terminal state is not included in the count.
        
        Returns:
            int: The number of steps in the episode.
        """
        return self.num_moves
    
    def __getitem__(self, index: int) -> tuple[int, int, float]:
        """Return the position, move, and reward at the given index.

        Returns the tuple pos_t, move_t, reward_t for the step at the given
        index or t in the episode. The terminal position is not included.

        Args:
            index (int): The index of the step.

        Returns:
            tuple[int, int, float]: The position, move, and reward.
        """
        return self.positions[index], self.moves[index], self.rewards[index]

    def add_move(
            self,
            position: tuple[int, int],
            move: str,
            reward: float
    ) -> None:
        """Add a move to the episode.

        Args:
            position (tuple[int, int]): The position moved from.
            move (str): The move made.
            reward (float): The reward received.
        """
        self.positions.append(position)
        self.moves.append(move)
        self.rewards.append(reward)

    def __str__(self) -> str:
        """Return a string representation of the episode."""
        epsilon = f"{self.epsilon:.4f}" if isinstance(self.epsilon, float) else self.epsilon
        s = f"Ep.{str(self.ep_num):>4} (Epsilon: {self.epsilon}) -"
        s += f" Steps:{str(self.num_moves):>5}"
        s += f" Reward:{str(self.total_reward):>4}"
        s += f" First and last position: {self.positions[0]} {self.last_pos}"
        return s
