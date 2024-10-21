"""Implementation of agent."""
from typing import Optional, Literal, Callable, Union, Hashable

from environment import GridWorld, Point, MoveSet, Move
from config import GridWorldPrefs
from .learning_model import LearningModel
from .episode import Episode


class Robot:
    """Robot class.

    This class represents the agent that will explore the environment. It
    interacts with the environment and learns from the feedback. The robot can
    explore the environment using the learning model, and follow its policy to
    reach the goal.
    """

    def __init__(
            self,
            environment: GridWorld | GridWorldPrefs,
            learning_model: LearningModel
    ) -> None:
        """Initialize the robot.

        Args:
            environment (GridWorld | GridWorldPrefs): The environment.
            learning_model (LearningModel): The learning model
        """
        if isinstance(environment, GridWorldPrefs):
            environment = GridWorld(environment)
        self._environment: GridWorld = environment
        self._position: Point = environment.start_position
        self._move_set: MoveSet = environment.move_set
        if not learning_model.initialized:
            learning_model.init_state_action_space(
                environment.state_space_shape, self._move_set)
        self._learning_model: LearningModel = learning_model
        self._learning_enabled = True

        self._goal_reached = False
        self._ep_history: list[Episode] = []
        self._abort_exploration = False

    @property
    def cur_episode(self) -> Episode | None:
        """Return the current episode."""
        if not self._ep_history:
            return None
        return self._ep_history[-1]

    @property
    def cur_episode_num(self) -> int:
        """Return the current episode number."""
        ep_num = len(self._ep_history) - 1
        return 0 if ep_num < 0 else ep_num

    @property
    def episodes(self) -> list[Episode]:
        """Return the recorded episodes."""
        return self._ep_history[:]

    @property
    def goal_reached(self) -> bool:
        """Return whether the goal has been reached."""
        return self._goal_reached

    @property
    def position(self) -> Point:
        """Return the position of the robot.

        Returns:
            tuple[int, int]: The x, y position of the robot
        """
        return self._position

    @property
    def learning_model(self) -> LearningModel:
        """Return the policy of the robot."""
        return self._learning_model

    def make_move(
            self,
            move: Optional[Union[Move, Hashable]] = None
    ) -> None:
        """Move the robot with the given move.

        Move the agent according to the given move. If no move is given, the
        best move from the policy is used. The agent will recive feedback from
        the environment and learn from it.

        Args:
            move (Move, optional): The move to make.
        """
        # Validate input
        if move is None:
            move = self.get_best_move()
        elif move not in self.move_set:
            raise ValueError("Invalid direction.")

        feedback = self._environment.make_move(self.position, move)
        self.cur_episode.add_move(
            tuple(self.position),
            move.KEY,
            feedback.REWARD
        )
        self._goal_reached = feedback.GOAL_REACHED
        if self._learning_enabled:
            self._learning_model.on_step_update(
                tuple(self.position),
                move,
                feedback.REWARD,
                feedback.NEW_POS)
        self._position = feedback.NEW_POS

    def _target_position(self, move) -> tuple[int, int]:
        """Calculate target position when making the move."""
        dx, dy = self._move_set[move]
        x, y = self.position
        return x + dx, y + dy

    def get_best_move(self) -> Move:
        """Get the best move from the policy."""
        return self._learning_model.get_action(self.position)

    def move_to_goal(
            self,
            abort: Callable[[], bool] | None = None,
            max_steps: int | None = None
    ) -> None:
        """Move the robot to the goal."""
        abort = abort if abort is not None else lambda: False
        i = 0
        while not self._goal_reached and not abort() and (
                max_steps is None or i < max_steps):
            self.make_move()  # Best move from policy
            i += 1

    def explore(
            self,
            stop_on_convergence: bool = True,
            random_reset: bool = True,
            on_new_episode: Callable[[Episode], None] = None,
            abort: Callable[[], bool] | None = None,
            ep_max: int | None = None,
            step_max: int | None = None
    ) -> None:
        """Explore the environment.

        Explores the environment using the learning model, until convergence,
        the maximum number of episodes is reached, or the exploration is
        aborted by the user.

        Args:
            stop_on_convergence (bool): Stop exploration when the policy
                converges.
            random_reset (bool): Reset the robot to a random position at the
                start of each episode.
            on_new_episode (Callable): Callback function to call when a new
                episode is started.
            abort (Callable): Function that returns True when the exploration
                should be cancelled.
        """
        start_pos = 'random' if random_reset else 'start'
        abort = abort if abort is not None else lambda: False
        if self._learning_enabled:
            self._learning_model.on_start_explore()
        i = 0
        while ((stop_on_convergence is False
                or not self._learning_model.converged)
                and not abort()
                and (ep_max is None or i < ep_max)):
            self._start_new_episode(start_pos, on_new_episode)
            self.move_to_goal(abort, step_max)
            i = i + 1

    def simulate_episodes(
            self,
            num_episodes: int,
            random_reset: bool = True,
            on_new_episode: Callable[[Episode], None] = None,
            abort: Callable[[], bool] | None = None,
            step_max: int = 100
    ) -> None:
        """Simulate the given number of episodes.

        Simulates the given number of episodes without learning. The robot
        will follow the policy to reach the goal. The learning model will not
        be updated.

        Args:
            num_episodes (int): The number of episodes to simulate.
            random_reset (bool): Reset the robot to a random position at the
                start of each episode.
            on_new_episode (Callable): Callback function to call when a new
                episode is started.
            abort (Callable): Function that returns True when the simulation
                should be cancelled.
            step_max (int): The maximum number of steps to simulate.
        """
        self._learning_enabled = False
        # Set epsilon to 0 for total exploitation
        epsilon_bac = self._learning_model.epsilon
        self._learning_model.epsilon = 0

        self._ep_history = []
        self.explore(
            stop_on_convergence=False,
            random_reset=random_reset,
            ep_max=num_episodes,
            on_new_episode=on_new_episode,
            abort=abort,
            step_max=step_max
        )
        self._learning_model.epsilon = epsilon_bac
        self._learning_enabled = True

    def _start_new_episode(
            self,
            start_pos: Point | Literal['start', 'random'] = 'random',
            callback: Callable[[Episode], None] = None
    ) -> Episode:
        """Start a new episode."""
        # Store the last position of the previous episode
        if self.cur_episode is not None:
            self.cur_episode.last_pos = self.position
            if callback is not None:
                callback(self.cur_episode)
            if self._learning_enabled:
                self._learning_model.on_episode_end(self.cur_episode)

        # Set init values for the new episode
        ep = Episode(self.cur_episode_num + 1, self._learning_model.epsilon)
        self._ep_history.append(ep)
        self._goal_reached = False
        # Set the start position
        if start_pos == 'random':
            self._position = self._environment.get_random_location()
        elif start_pos == 'start':
            self._position = self._environment.start_position
        else:
            self._position = start_pos

        return ep

    def greedy_path(
            self,
            from_pos: Point | Literal['start', 'random'] = 'start'
    ) -> list[tuple[int, int]]:
        """Find the greedy path.

        Find the path to the goal using the policy. The robot will follow the
        policy to reach the goal.

        Args:
            from_pos (tuple[int, int] | str): The starting position. If 'start'
                the robot will start at the initial position. If 'random' the
                robot will start at a random position.
        """
        # Set epsilon to 0 for total exploitation
        epsilon_bac = self._learning_model.epsilon
        self._learning_model.epsilon = 0

        ep = self._start_new_episode(from_pos)
        self.move_to_goal()
        self._learning_model.epsilon = epsilon_bac
        return ep.positions
