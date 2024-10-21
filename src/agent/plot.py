"""Plotting functions to visualize the training process.

The functions in this module are used to visualize the training process
of the Q-learning algorithm. The functions are used to plot the rewards,
steps, epsilon, state heatmap, state action heatmap, and Q-table heatmap.

Methods:
    plot_exploration_data: Make and display plots of exploration data.
    plot_rewards: Plot the total reward for selected episodes (scatterplot).
    plot_steps: Plot the number of steps for selected episodes (scatterplot).
    plot_epsilon: Plot the epsilon decay rate for selected episodes.
    plot_state_heatmap: Plot the state visited heatmap for selected episodes.
    plot_q_table_heatmap: Plot heatmap of Q-values.
    plot_state_action_heatmap: Plot frequency of taking each action for all
                               states as a heatmap for selected episodes.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.tri import Triangulation
from numpy.typing import NDArray
from typing import Literal, Optional

from .episode import Episode


def plot_exploration_data(
        episodes: list[Episode],
        q_table: Optional[dict] = None):
    """Make and display plots of exploration data.

    This function makes and displays some reasonable plots of the exploration
    data. The plots include rewards, steps, epsilon, state heatmap, state
    action heatmap, and Q-table heatmap.

    A pyplot function, like .show(), must be called after this function to
    render the plots.

    Args:
        episodes (list): The list of episodes.
        q_table (Optional[dict]): The Q-table. Defaults to None,
            in which case the Q-table heatmap is not plotted
    """
    # Exploration plots
    ep_n_mid = len(episodes) // 2
    fig1, axs1 = plt.subplots(3, 2, figsize=(10, 10))
    plot_rewards(episodes, plot_to=(fig1, axs1[0, 0]))
    plot_steps(episodes, plot_to=(fig1, axs1[0, 1]))
    plot_rewards(episodes, ep_n_mid, type='bar', plot_to=(fig1, axs1[1, 0]))
    plot_steps(episodes, ep_n_mid, type='bar', plot_to=(fig1, axs1[1, 1]))
    if episodes[0].epsilon is not None:
        plot_epsilon(episodes, plot_to=(fig1, axs1[2, 0]))
    else:
        axs1[2, 0].axis("off")
    axs1[2, 1].axis("off")
    fig1.suptitle("Exploration plots")
    fig1.tight_layout()

    # Exploration heatmaps
    fig2, axs2 = plt.subplots(1, 2, figsize=(16, 8))
    plot_state_heatmap(episodes, plot_to=(fig2, axs2[0]))
    plot_state_action_heatmap(episodes, plot_to=(fig2, axs2[1]))
    fig2.suptitle("Exploration heatmaps")
    fig2.tight_layout()

    # Q-table heatmap
    if q_table is not None:
        fig3, axs3 = plt.subplots(1, 1, figsize=(8, 8))
        plot_q_table_heatmap(q_table, plot_to=(fig3, axs3))
        fig3.suptitle("Q-table heatmap")
        fig3.tight_layout()


def _transform_relative_index(index, length,
                              dir: Literal['from', 'to'] = 'from'):
    """Transform a relative index to an absolute index."""
    if index is None:
        return 0 if dir == 'from' else length
    if index < 0:
        index += length
        return 0 if index < 0 else index
    if index >= length:
        return length - 1
    return index


def plot_rewards(
        episodes: list[Episode],
        start_ep: int | None = None,
        end_ep: int | None = None,
        type: Literal['scatter', 'bar'] = 'scatter',
        plot_to=None
) -> plt.Figure:
    """Plot the rewards."""
    if not plot_to:
        fig, axs = plt.subplots()
    else:
        fig, axs = plot_to

    rewards = [ep.total_reward for ep in episodes[start_ep:end_ep]]
    start_ep_nr = _transform_relative_index(start_ep, len(episodes))
    end_ep_nr = _transform_relative_index(end_ep, len(episodes), dir='to')

    if type == 'scatter':
        axs.scatter(range(start_ep_nr, end_ep_nr), rewards, s=5)
    elif type == 'bar':
        axs.bar(range(start_ep_nr, end_ep_nr), rewards, width=1)
    axs.set_title(f"Rewards ep. {start_ep_nr} to {end_ep_nr - 1} (incl.)")
    axs.set_xlabel("Episode")
    axs.set_ylabel("Reward")

    return fig


def plot_steps(
        episodes: list[Episode],
        start_ep: int | None = None,
        end_ep: int | None = None,
        type: Literal['scatter', 'bar'] = 'scatter',
        plot_to=None
) -> plt.Figure:
    """Plot the steps."""
    if not plot_to:
        fig, axs = plt.subplots()
    else:
        fig, axs = plot_to

    steps = [ep.num_moves for ep in episodes[start_ep:end_ep]]
    start_ep_nr = _transform_relative_index(start_ep, len(episodes))
    end_ep_nr = _transform_relative_index(end_ep, len(episodes), dir='to')

    if type == 'bar':
        axs.bar(range(start_ep_nr, end_ep_nr), steps, width=1)
    elif type == 'scatter':
        axs.scatter(range(len(steps)), steps, s=5)
    axs.set_title(f"Steps ep. {start_ep_nr} to {end_ep_nr - 1} (incl.)")
    axs.set_xlabel("Episode")
    axs.set_ylabel("Steps")

    return fig


def plot_epsilon(episodes: list[Episode], plot_to=None):
    """Plot the epsilon."""
    if not plot_to:
        fig, axs = plt.subplots()
    else:
        fig, axs = plot_to

    epsilons = [ep.epsilon for ep in episodes]
    axs.plot(epsilons)
    axs.set_title("Epsilon")
    axs.set_xlabel("Episode")
    axs.set_ylabel("Epsilon")
    return fig


def _count_pos_frequency(
        episodes: list[Episode],
        start_ep: int | None = None,
        end_ep: int | None = None
) -> NDArray[np.int64]:
    """Count the position frequency."""
    max_x, max_y = 0, 0
    for ep in episodes[start_ep:end_ep]:
        for x, y in ep.positions:
            max_x = max(x, max_x)
            max_y = max(y, max_y)

    pos_freq = np.zeros((max_y + 1, max_x + 1), dtype=int)
    for ep in episodes[start_ep:end_ep]:
        for x, y in ep.positions:
            pos_freq[y, x] += 1

    return pos_freq


def plot_state_heatmap(
        episodes: list[Episode],
        start_ep: int | None = None,
        end_ep: int | None = None,
        plot_to: tuple = None,
        move_keys_cw: list[str] = ['N', 'E', 'S', 'W']
) -> plt.Figure:
    """Plot the state heatmap."""
    pos_freq = _count_pos_frequency(episodes, start_ep, end_ep)
    if not plot_to:
        fig, axs = plt.subplots()
    else:
        fig, axs = plot_to

    cax = axs.imshow(pos_freq, cmap='viridis', origin='upper',
                     interpolation='nearest')
    # Add a color bar as a legend
    cbar = fig.colorbar(cax)
    cbar.set_label('Frequency')

    # Add value labels
    max_val = np.max(pos_freq)
    mid_val = max_val // 2
    for (r, c), value in np.ndenumerate(pos_freq):
        axs.text(c, r, int(value), ha='center', va='center',
                 color='w' if value < mid_val else 'k')
    # Set plot properties
    start_ep_nr = _transform_relative_index(start_ep, len(episodes))
    end_ep_nr = _transform_relative_index(end_ep, len(episodes), dir='to')
    axs.set_title(f"State heatmap ep. {start_ep_nr} " +
                  f"to {end_ep_nr - 1} (incl.)")
    axs.set_xlabel("X")
    axs.set_ylabel("Y")
    return fig


def plot_q_table_heatmap(
        q_table: dict[tuple[int, int], dict[str, float]],
        plot_to: tuple[plt.Figure, plt.Axes] | None = None,
        move_keys_cw: list[str] = ['N', 'E', 'S', 'W']
) -> plt.Figure:
    """Plot the Q-table heatmap.

    Plots to new figure, or to the provided plot_to fig, ax tuple.
    Move keys are assumed to be in clockwise order, starting from
    the top. There must be exactly 4 move keys. The move keys must
    correspond to the keys for moves in the Q-table.

    Args:
        q_table (dict): The Q-table. The keys are (x, y) positions and the
            values are dictionaries with move keys as keys and Q-values as
            values.
        plot_to (tuple): Optional pyplot (fig, ax) tuple to plot to.
        move_keys_cw (list): The clockwise move keys.

    Returns:
        plt.Figure: The figure with the heatmap.
    """
    x_max, y_max = max(q_table.keys())
    rows, cols = y_max + 1, x_max + 1
    val_nesw = [
        np.zeros((rows, cols))
        for _ in range(len(move_keys_cw))
    ]
    for (x, y), q_vals in q_table.items():
        for i, action in enumerate(move_keys_cw):
            val_nesw[i][y, x] = q_vals[action]
    title = "Q-table heatmap"
    cbar_label = "Q-value"
    return _plot_triag_heatmap(val_nesw, title, cbar_label, plot_to)


def plot_state_action_heatmap(
        episodes: list[Episode],
        start_ep: int | None = None,
        end_ep: int | None = None,
        plot_to: tuple = None,
        move_keys_cw: list[str] = ['N', 'E', 'S', 'W']
) -> plt.Figure:
    """Plot the state action heatmap."""
    # Get the maximum x and y values
    x_max, y_max = 0, 0
    for ep in episodes[start_ep:end_ep]:
        for (x, y), move, _ in ep:
            x_max = max(x, x_max)
            y_max = max(y, y_max)

    # Count move frequency
    pos_move_freq = np.zeros((y_max + 1, x_max + 1, len(move_keys_cw)),
                             dtype=int)
    keymap = {key: i for i, key in enumerate(move_keys_cw)}
    for ep in episodes[start_ep:end_ep]:
        for (x, y), move, _ in ep:
            pos_move_freq[y, x, keymap[move]] += 1

    # Prep data for heatmap plot
    val_nesw = [pos_move_freq[:, :, i] for i in range(len(move_keys_cw))]
    title = "State action heatmap for ep. "
    start_ep_nr = _transform_relative_index(start_ep, len(episodes))
    end_ep_nr = _transform_relative_index(end_ep, len(episodes), dir='to')
    title += f"{start_ep_nr} to {end_ep_nr - 1} (incl.)"
    cbar_label = "Frequency"
    return _plot_triag_heatmap(val_nesw, title, cbar_label, plot_to)


def _plot_triag_heatmap(val_nesw, title, cbar_label, plot_to):
    """Plot heatmap with four triangles per cell.

    This method is partially sourced from:
    https://stackoverflow.com/questions/66048529/
    """
    min_val = np.min(val_nesw)
    max_val = np.max(val_nesw)
    mid_val = max_val / 2
    # Plot the heatmap
    if not plot_to:
        fig, ax = plt.subplots()
    else:
        fig, ax = plot_to
    cols, rows = val_nesw[0].shape
    triangul = _triangulation_heatmap(cols, rows)
    norms = [plt.Normalize(min_val, max_val) for _ in range(4)]
    imgs = [
        ax.tripcolor(trig, val.ravel(), cmap='viridis',
                     norm=norm, ec='white')
        for trig, val, norm in zip(triangul, val_nesw, norms)
    ]
    # Add value labels
    for val, dir in zip(val_nesw, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for x in range(cols):
            for y in range(rows):
                v = val[y, x]
                v_str = f"{v:.2f}" if isinstance(
                    v, np.floating) else str(v)
                ax.text(
                    x + 0.3 * dir[1],
                    y + 0.3 * dir[0],
                    v_str,
                    color='k' if v > mid_val else 'w',
                    ha='center',
                    va='center'
                )
    # Add color bar
    cbar = fig.colorbar(imgs[0], ax=ax)
    cbar.set_label(cbar_label)

    # Set plot properties
    ax.set_title(title)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    return fig


def _triangulation_heatmap(M, N) -> list[Triangulation]:
    """Return the triangulation for the heatmap.

    This method is sourced from:
    https://stackoverflow.com/questions/66048529/
    """
    # Grid of vertices for each square in heatmap
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))
    # Grid of centers for each square in heatmap
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))

    # Flatten the vertex and center arrays to concacenated lists
    # of x and y coordinates
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    # Calculate the starting index for centers
    cstart = (M + 1) * (N + 1)

    # Lists to store triangle indices for each direction
    triN, triE, triS, triW = [], [], [], []
    for n in range(N):
        for m in range(M):
            # Calculate the base index for the current (m, n)
            base_index = m + n * (M + 1)
            # Calculate the center index
            center_index = cstart + m + n * M

            # North triangle
            triN.append((
                base_index,
                base_index + 1,
                center_index
            ))
            # East triangle
            triE.append((
                base_index + 1,
                base_index + (M + 1) + 1,
                center_index
            ))
            # South triangle
            triS.append((
                base_index + (M + 1) + 1,
                base_index + (M + 1),
                center_index
            ))
            # West triangle
            triW.append((
                base_index + (M + 1),
                base_index,
                center_index
            ))

    # Return list of triangulation objects for N, E, S, W
    # triangles dividing each cell of heatmap
    return [
        Triangulation(x, y, triangles)
        for triangles in [triN, triE, triS, triW]
    ]
