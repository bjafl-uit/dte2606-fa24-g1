"""Module to simulate the robot in a pygame window."""
import pygame
from pygame.locals import KEYDOWN, K_ESCAPE, QUIT, Rect
from dataclasses import dataclass
from typing import Callable
import threading
from queue import Empty
import time

from config import ASSETS_PATH
from agent.episode import Episode

from .slider import Slider
from .play_btn import PlayPauseButton


@dataclass
class RobotState:
    """Robot state class."""

    pos: tuple[int, int]
    reward: float
    epsilon: float


class RoboSim:
    """Class to simulate the robot in a pygame window."""

    GREEN = pygame.Color(0, 255, 0)
    BLACK = pygame.Color(0, 0, 0)
    WHITE = pygame.Color(255, 255, 255)
    LIGHT_GRAY = pygame.Color(211, 211, 211)
    SLATE_GRAY = pygame.Color(136, 136, 136)
    FONT_PATH = ASSETS_PATH / 'CONSOLA.TTF'

    def __init__(
            self,
            map_shape: tuple[int, int],
            map_colors: dict[str, tuple[int, int, int]],
            get_episode: Callable[[], Episode],
            update_main_window: Callable[[], None],
            screen_width: tuple[int, int] = 1000
    ):
        """Initialize the simulator."""
        pygame.init()
        self._screen_size = (screen_width, int(screen_width * 1.05))
        self._play_surface = pygame.display.set_mode(self._screen_size)
        self._simulator_speed = 0  # 0 is 1 update per frame
        self._fps = 50
        self._running = False
        self._updates_skipped = 0

        self._map_margin = 70
        self._robot_size = 20
        self._map_shape = map_shape
        self._map_colors = map_colors
        self._map_surface = pygame.Surface(self._screen_size)
        self._map_cell_w = ((self._screen_size[0] - 2 * self._map_margin)
                            // self._map_shape[0])
        self._map_cell_h = self._map_cell_w
        self._color_window_bg = self.BLACK
        self._color_map_cell = self.LIGHT_GRAY
        self._color_map_border = self.BLACK
        self._color_map_font = self.WHITE
        self._render_map()
        self._stats_font_size = 18
        self._stats_font = pygame.font.Font(str(self.FONT_PATH),
                                            self._stats_font_size)

        self._current_ep_state: tuple[int, int] | None = (0, 0)
        self._robot_states: list[list[RobotState]] = []
        self._pop_episode_data = get_episode
        self._all_episodes_loaded = False
        self._lock = threading.Lock()
        self._episode_loader_thread = threading.Thread(
            target=self._episode_loader)
        self._episode_loader_thread.start()
        self._update_main_window = update_main_window

        self._n_hist_draw = 100
        self._history_colors = self._generate_red_gradient(self._n_hist_draw)

        y_offset = 40
        slider_w = screen_width // 2
        slider_h = 20
        play_btn_wh = 50
        screen_h = self._screen_size[1]
        slider_x = screen_width - slider_w - 20
        state_slider_y = screen_h - slider_h - y_offset
        n_eps, n_states = self._get_n_ep_state()
        self._state_slider = Slider(
            slider_x,
            state_slider_y,
            slider_w,
            slider_h,
            min_value=0,
            max_value=n_states,
            start_val=0)
        self._ep_slider = Slider(
            slider_x,
            state_slider_y - slider_h - 10,
            slider_w,
            slider_h,
            min_value=0,
            max_value=n_eps,
            start_val=0)
        play_y = screen_h - play_btn_wh - y_offset
        self._play_btn = PlayPauseButton(
            10, play_y,
            play_btn_wh,
            play_btn_wh)
        speed_slider_w = slider_w // 2
        speed_slider_x = 10 + play_btn_wh + 10
        speed_slider_y = play_y + play_btn_wh // 2 - slider_h // 2
        self._speed_slider = Slider(
            speed_slider_x,
            speed_slider_y,
            speed_slider_w, slider_h,
            min_value=-60,
            max_value=60,
            start_val=self._simulator_speed)
        self._draw_text("Speed", 20, (speed_slider_x+10, speed_slider_y + 20),
                        self.BLACK, self._map_surface, self._stats_font)
        self._draw_text("Episode", 20, (slider_x-70, state_slider_y - 30),
                        self.BLACK, self._map_surface, self._stats_font)
        self._draw_text("State", 20, (slider_x-70, state_slider_y),
                        self.BLACK, self._map_surface, self._stats_font)

        pygame.display.set_caption('Robosim')  # TODO

    @property
    def cur_state(self):
        """Return the current state."""
        e, s = self._current_ep_state
        if (e < 0 or e >= len(self._robot_states)
                  or s >= len(self._robot_states[e])):
            return None
        return self._robot_states[e][s]
    
    def _get_n_ep_state(self, ep_num=None):
        """Return the number of episodes and states in gven episode."""
        if len(self._robot_states) == 0:
            return 0, 0
        ep_num = self._current_ep_state[0] if ep_num is None else ep_num
        return len(self._robot_states), len(self._robot_states[ep_num])

    def add_episode_data(self, episode: Episode):
        """Add episode data to the simulator."""
        self._episode_queue.append(episode)

    def run(self, on_sim_done):
        """Run the simulation."""
        try:
            self._main_loop()
        except InterruptedError:
            pass
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False
            pygame.quit()
            self._on_sim_done()
            # TODO

    def _main_loop(self):
        self._running = True
        fps_clock = pygame.time.Clock()
        while self._running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == QUIT:
                    raise InterruptedError
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        raise KeyboardInterrupt
                self._ep_slider.handle_event(event)
                self._state_slider.handle_event(event)
                self._speed_slider.handle_event(event)
                self._play_btn.handle_event(event)
            # Draw background
            self._play_surface.blit(self._map_surface, (0, 0))

            self._update_main_window()

            # Update simulator speed
            self._simulator_speed = int(self._speed_slider.value)
            # Update robot state
            n_eps, n_states = self._get_n_ep_state()
            self._ep_slider.max_val = n_eps - 1
            self._state_slider.max_val = n_states - 1
            if self._play_btn.get_state():
                # Get next robot state
                self._next_robot_state()
                self._ep_slider.value = self._current_ep_state[0]
                self._state_slider.value = self._current_ep_state[1]
                self._ep_slider.disabled = True
                self._state_slider.disabled = True
            else:
                self._ep_slider.disabled = False
                self._state_slider.disabled = False
                self._update_robot_state(
                    int(self._ep_slider.value),
                    int(self._state_slider.value),
                )
            # Draw robot
            if self.cur_state is not None:
                self._draw_robot_history()
                self._draw_robot()
                self._draw_stats()

            # Draw slider
            self._ep_slider.draw(self._play_surface)
            self._state_slider.draw(self._play_surface)
            self._speed_slider.draw(self._play_surface)
            # Draw btn
            self._play_btn.draw(self._play_surface)

            # Refresh the screen.
            pygame.display.flip()
            fps_clock.tick(self._fps)

    def _draw_robot(self):
        """Draw the robot."""
        dim = self._robot_size
        x, y = self._map_cell_pos(*self.cur_state.pos, center=True)
        x -= dim // 2
        y -= dim // 2
        self._draw_rect((dim, dim), (x, y), self.GREEN, (1, self.BLACK))

    def _draw_robot_history(self):
        """Draw the robot history."""
        if self.cur_state is None:
            return
        e, s = self._current_ep_state
        n_eps, _ = self._get_n_ep_state()
        states = self._robot_states[e][max(0, s-self._n_hist_draw):s]
        n_colors = max(min(n_eps, self._n_hist_draw), 5)
        colors = reversed(self._generate_red_gradient(n_colors)[:len(states)])
        for state, color in zip(states, colors):
            dim = self._robot_size
            x, y = self._map_cell_pos(*state.pos, center=True)
            x -= dim // 2
            y -= dim // 2
            self._draw_rect((dim, dim), (x, y), color, (1, self.BLACK))

    def _generate_red_gradient(self, steps):
        gradient_colors = []
        for i in range(steps):
            R = 200 - (200 // steps) * i  # From 200 down to 0
            G = 100 - (100 // steps) * i  # Muted green
            B = 100 - (100 // steps) * i  # Muted blue
            gradient_colors.append((R, G, B))
        return gradient_colors

    def _render_map(self):
        """Render the map."""
        self._map_surface.fill(self._color_window_bg)
        cell_dim = (self._map_cell_w, self._map_cell_h)
        # Draw map grid
        for c in range(self._map_shape[0]):
            for r in range(self._map_shape[1]):
                self._draw_rect(
                    cell_dim,
                    self._map_cell_pos(c, r),
                    self._map_colors[(c, r)],
                    (1, self._color_map_border),
                    self._map_surface
                )
        # Draw axes
        font_size = self._map_margin // 3
        col_text_margin = self._map_margin - font_size * 1.5
        for c in range(self._map_shape[0]):
            s = str(c)
            x = self._map_cell_pos(c, 0, True)[0] - font_size * len(s) // 2
            self._draw_text(s, font_size, (x, col_text_margin),
                            self._color_map_font, self._map_surface)
        col_labels = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for r in range(self._map_shape[1]):
            s = col_labels[r]
            y = self._map_cell_pos(0, r, True)[1] - font_size * len(s) // 2
            self._draw_text(s, font_size, (col_text_margin, y),
                            self._color_map_font, self._map_surface)

        # Fill bottom color
        y = self._map_cell_pos(0, self._map_shape[1])[1] + 20
        self._draw_rect((self._screen_size[0], self._screen_size[1] - y),
                        (0, y), self.SLATE_GRAY, draw_on=self._map_surface)

    def _map_cell_pos(self, c_nr, r_nr, center: bool = False):
        """Return the position of the cell in the map."""
        x, y = (c_nr * self._map_cell_w + self._map_margin,
                r_nr * self._map_cell_h + self._map_margin)
        if center:
            x += self._map_cell_w // 2
            y += self._map_cell_h // 2
        return x, y

    def _draw_rect(self, dimension, position, fill, border=None, draw_on=None):
        w, h = dimension
        x, y = position
        if draw_on is None:
            draw_on = self._play_surface
        if border:
            bw, bc = border
            pygame.draw.rect(draw_on, bc,
                             Rect(x - bw, y - bw, w + 2 * bw, h + 2 * bw))
        pygame.draw.rect(draw_on, fill, Rect(x, y, w, h))

    def _draw_text(self, text, font_size, position,
                   color=None, draw_on=None, font=None):
        if color is None:
            color = self.BLACK_COLOR,
        if draw_on is None:
            draw_on = self._play_surface
        if font is None:
            font = pygame.font.Font(None, font_size)
        text_surface = font.render(text, True, color)
        draw_on.blit(text_surface, position)

    def _draw_stats(self, box_height=20):
        """Draw the stats."""
        dim = (self._screen_size[0], box_height)
        pos = (0, self._screen_size[1] - box_height)
        self._draw_rect(dim, pos, self.WHITE)
        text_surface = self._stats_font.render(self._stats_text(),
                                               True, self.BLACK)
        self._play_surface.blit(
            text_surface,
            (pos[0] + self._stats_font_size,
             pos[1] + (box_height - self._stats_font_size) // 2)
        )

    def _stats_text(self):
        """Return the stats text."""
        n_eps, n_states = self._get_n_ep_state()
        e, s = self._current_ep_state
        txt = f"Episode: {e:>5} / {n_eps-1:>5} | "
        txt += f"Total Reward: {self.cur_state.reward:>9} | "
        txt += f"Steps: {s:>5} / {n_states-1:>5} | "
        if self.cur_state.epsilon is not None:
            txt += f"Epsilon: {self.cur_state.epsilon:.4f}"

    def _next_robot_state(self):
        """Return the next robot state."""
        if self.cur_state is None:
            return

        if self._simulator_speed < 0:
            self._updates_skipped -= 1
        skip_draw = 0 if self._simulator_speed >= 0 else self._updates_skipped
        while skip_draw <= self._simulator_speed:
            self._updates_skipped = 0
            e, s = self._current_ep_state
            n_ep, n_steps = self._get_n_ep_state()
            if s + 1 < n_steps:
                self._current_ep_state = (e, s + 1)
            elif e + 1 < n_ep:
                self._current_ep_state = (e + 1, 0)
            skip_draw += 1

    def _update_robot_state(self, ep_num, state_num):
        """Update the robot state."""
        n_ep, n_steps = self._get_n_ep_state(ep_num)
        if ep_num < 0:
            ep_num = 0
        elif ep_num >= n_ep:
            ep_num = n_ep - 1
        if state_num < 0:
            state_num = 0
        elif state_num >= n_steps:
            state_num = n_steps - 1
        self._current_ep_state = (ep_num, state_num)

    def _episode_loader(self):
        while not self._all_episodes_loaded:
            try:
                with self._lock:
                    episode = self._pop_episode_data()
            except Empty:
                continue
            except ValueError:
                with self._lock:
                    self._all_episodes_loaded = True
                break
            self._load_episode_data(episode)
            time.sleep(0.003)

    def _load_episode_data(self, episode: Episode):
        """Load the episode data."""
        n_steps = len(episode.positions)
        reward = 0
        states = []
        for i in range(n_steps):
            if i > 0:
                reward += episode.rewards[i-1]
            pos = episode.positions[i]
            states.append(RobotState(
                pos=pos,
                reward=reward,
                epsilon=episode.epsilon
            ))
        with self._lock:
            self._robot_states.append(states)
