"""Main menu for the application.

Tkinter window that allows the user to set hyperparameters for the Q-learning
algorithm and start the exploration process. The user can also open a
simulation window to visualize the robot's exploration process and a plot
window to visualize the exploration data.

Classes:
    MainMenu: The main menu for the application.
"""

from tkinter import Tk, Label, Button, Radiobutton, Checkbutton, IntVar
import threading
import time
from queue import Queue
import matplotlib.pyplot as plt
from typing import Literal

from ._frames import HyperparamsFrame, ExploreParamsFrame
from config import ExploreParamsDefaults, GridWorldPrefs
from environment import GridWorld
from agent import Robot, QLearning, DynamicEpsilon, Episode, MonteCarlo
from agent.plot import plot_exploration_data
from simulation import RoboSim


class MainMenu(Tk):
    """Main menu for the application.

    Tkinter window that lets the user control the exploration and visualization
    of the Reinforcement Learning agent.
    """

    def __init__(
            self,
            grid_world_prefs: GridWorldPrefs,
            explore_params_defaults: ExploreParamsDefaults
    ) -> None:
        """Initialize the main menu.

        Args:
            grid_world_prefs: The preferences for the grid world environment.
            explore_params_defaults: The default exploration parameters.
        """
        super().__init__()

        self.title("Reinforcement Learning")
        row = 0
        # Learning model selection
        self._learning_model: Literal['Q', 'MC'] = 'Q'
        self._lbl_model = Label(self, text="Learning Model: ")
        self._lbl_model.grid(row=row, column=0, columnspan=2, padx=5, pady=5)
        row += 1
        self._radio_q = Radiobutton(
            self,
            text="Q-Learning",
            value='Q',
            variable=self._learning_model,
            command=lambda: self._select_learning_model('Q')
        )
        self._radio_q.grid(row=row, column=0, padx=5, pady=5, sticky='e')
        self._radio_mc = Radiobutton(
            self,
            text="Monte Carlo",
            value='MC',
            variable=self._learning_model,
            command=lambda: self._select_learning_model('MC')
        )
        self._radio_mc.grid(row=row, column=1, padx=5, pady=5, sticky='w')

        row += 1
        # Hyperparameters and exploration parameters
        self._hyperparams = HyperparamsFrame(self, 15, explore_params_defaults)
        self._hyperparams.grid(row=row, column=0, padx=5, pady=5)

        self._explore_params_frame = ExploreParamsFrame(
            self,
            18,
            explore_params_defaults)
        self._explore_params_frame.grid(row=row, column=1, padx=5, pady=5)

        # Simulate greedy checkbox
        row += 1
        self._chk_greedy_checked = IntVar(value=0)
        self._chk_greedy = Checkbutton(
            self,
            text="Greedy Simulation of Policy",
            variable=self._chk_greedy_checked
        )
        self._chk_greedy.grid(row=row, column=0, columnspan=2,
                              padx=5, pady=5)
        self._chk_greedy.config(state='disabled')

        # Start and stop buttons
        row += 1
        self._btn_start = Button(self, text="Start",
                                 command=self._start_explore)
        self._btn_start.grid(row=row, column=0, padx=5, pady=5, sticky='e')
        self._btn_stop = Button(self, text="Stop",
                                command=self._cancel_explore)
        self._btn_stop.config(state='disabled')
        self._btn_stop.grid(row=row, column=1, padx=5, pady=5, sticky='w')

        row += 1
        # Status label
        self._lbl_status = Label(self, text="Ready")
        self._lbl_status.grid(row=row, column=0, columnspan=2, padx=5, pady=5)

        row += 1
        self._btn_open_sim = Button(self, text="Open Simulation",
                                    command=self._open_sim)
        self._btn_open_sim.grid(row=row, column=0, padx=5, pady=5, sticky='e')
        self._btn_open_sim.config(state='disabled')
        self._btn_open_plot = Button(self, text="Open Plot",
                                     command=self._open_plot)
        self._btn_open_plot.grid(row=row, column=1, padx=5, pady=5, sticky='w')
        self._btn_open_plot.config(state='disabled')

        # Thread handling
        self._lock = threading.Lock()
        self._exploration_thread = None
        self._exploration_running = False
        self._abort_exploration = False
        self._update_status_freq = 100  # Every n episodes
        self._plot_thread = None
        self._plot_open = False

        # GridWorld init
        self._grid_world = GridWorld(grid_world_prefs)
        self._cell_colors = {}
        for y, row in enumerate(grid_world_prefs.MAP):
            for x, cell in enumerate(row):
                self._cell_colors[(x, y)] = grid_world_prefs.MAP_COLORS[cell]

        # Episode queue and last simulation
        self._episode_sim_queue = Queue()
        self._robo_sim = None
        self._robot_last_run: Robot | None = None
        self._sim_running = False

        self._radio_q.select()

    def _select_learning_model(self, val) -> None:
        self._learning_model = val
        if self._learning_model == 'Q':
            self._hyperparams.epsilon_enabled = True
            self._hyperparams.alpha_enabled = True
            self._explore_params_frame.conv_tol_enabled = True
        else:
            self._hyperparams.epsilon_enabled = False
            self._hyperparams.alpha_enabled = False
            self._explore_params_frame.conv_tol_enabled = False

    def update_status(self, episode) -> None:
        """Update the status label."""
        self._episode_sim_queue.put(episode)
        if episode.ep_num % self._update_status_freq == 0:
            self._update_status_label(episode)

    def _update_status_label(self, episode: Episode | str) -> None:
        if isinstance(episode, str):
            message = episode
        else:
            message = f"Ep.: {episode.ep_num:>6}, "
            message += f"Steps: {episode.num_moves:>6}, "
            if episode.epsilon is not None:
                message += f"Epsilon: {episode.epsilon:.4f}"
        with self._lock:
            self._lbl_status.config(text=message)
            self._time_last_update = time.time()

    def pop_episode(self) -> Episode:
        """Pop an episode from the sim queue."""
        if not self._exploration_running and self._episode_sim_queue.empty():
            raise ValueError("No episodes available.")
        return self._episode_sim_queue.get()

    def _on_explore_done(self, robot: Robot) -> None:
        self._robot_last_run = robot
        self._exploration_running = False
        if not self._sim_running:
            self._btn_start.config(state='normal')
            self._chk_greedy.config(state='normal')
        self._btn_stop.config(state='disabled')
        if self._abort_exploration:
            self._lbl_status.config(text="Exploration aborted.")
            self._abort_exploration = False
        else:
            self._lbl_status.config(text="Done!")
        if not self._plot_open:
            self._btn_open_plot.config(state='normal')

    def _cancel_explore(self) -> None:
        self._abort_exploration = True
        self._btn_stop.config(state='disabled')

    def _start_explore(self) -> None:
        self._update_status_label('Starting exploration...')
        self._abort_exploration = False
        self._exploration_running = True
        self._btn_start.config(state='disabled')
        self._btn_stop.config(state='normal')
        self._btn_open_sim.config(state='normal')
        self._btn_open_plot.config(state='disabled')
        self._chk_greedy.config(state='disabled')
        self._episode_sim_queue = Queue()
        self._exploration_thread = threading.Thread(
            target=self._run_exploration)
        self._exploration_thread.daemon = True
        self._exploration_thread.start()

    def _open_sim(self) -> None:
        self._btn_open_sim.config(state='disabled')
        self._btn_start.config(state='disabled')
        self._sim_running = True
        self.grab_set()
        self._robo_sim = RoboSim(self._grid_world.state_space_shape,
                                 self._cell_colors,
                                 self.pop_episode,
                                 self.update)
        self._robo_sim.run(self._on_sim_done)

    def _on_sim_done(self) -> None:
        self.after(200, self._update_main_after_sim_done)

    def _update_main_after_sim_done(self) -> None:
        self._sim_running = False
        self.grab_release()
        self._btn_open_sim.config(state='normal')
        self._lbl_status.config(text="Ready")
        if not self._exploration_running:
            self._btn_start.config(state='normal')

    def _update_main_window(self) -> None:
        self.update()
        if self._sim_running:
            self.after(100, self._update_main_window)

    def _open_plot(self) -> None:
        if self._plot_open:
            return
        self._plot_thread = threading.Thread(target=self._draw_plots)
        self._plot_open = True
        self._btn_open_plot.config(state='disabled')
        self._plot_thread.daemon = True
        self._plot_thread.start()

    def _draw_plots(self) -> None:
        plot_exploration_data(
            self._robot_last_run.episodes,
            self._robot_last_run.learning_model.get_unhashed_q_table()
        )
        plt.show()
        self._plot_open = False
        self._btn_open_plot.config(state='normal')

    def _run_exploration(self) -> None:
        eps_min, eps_max, eps_decay = self._hyperparams.decaying_epsilon_values
        ep_max = self._explore_params_frame.ep_max
        rtol, atol = self._explore_params_frame.conv_tol
        if self._chk_greedy_checked.get():
            epsilon = 0
            robot = self._robot_last_run
            robot.simulate_episodes(
                num_episodes=ep_max,
                random_reset=True,
                on_new_episode=self.update_status,
                abort=self._get_abort_flag
            )
        else:
            if self._hyperparams.decaying_epsilon:
                epsilon = DynamicEpsilon(eps_min, eps_max, eps_decay)
            else:
                epsilon = self._hyperparams.epsilon

            if self._learning_model == 'Q':
                learning_model = QLearning(
                    alpha=self._hyperparams.alpha,
                    gamma=self._hyperparams.gamma,
                    epsilon=epsilon,
                    conv_tolerance=(rtol, atol)
                )
            else:
                learning_model = MonteCarlo(
                    gamma=self._hyperparams.gamma
                )
            robot = Robot(self._grid_world, learning_model)
            robot.explore(
                ep_max=ep_max,
                on_new_episode=self.update_status,
                abort=self._get_abort_flag
            )

        self._on_explore_done(robot)

    def _get_abort_flag(self) -> bool:
        with self._lock:
            return self._abort_exploration
