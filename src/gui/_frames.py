"""Submodule with classes for the GUI frames.

This is a private module, only intended to be used by the main menu module.

Classes:
    EpsilonFrame: Frame for epsilon hyperparameters.
    HyperparamsFrame: Frame for general hyperparameters.
    ExploreParamsFrame: Frame for exploration parameters.
"""

from tkinter import Label, Entry, Frame, Checkbutton, Misc

from config import ExploreParamsDefaults


class EpsilonFrame(Frame):
    """Frame for setting epsilon hyperparameters.

    Attributes:
        lbl_epsilon (Label): Label for epsilon.
        lbl_epsilon_min (Label): Label for epsilon min.
        lbl_epsilon_max (Label): Label for epsilon max.
        lbl_epsilon_decay (Label): Label for epsilon decay rate.
        entry_epsilon (Entry): Entry for epsilon.
        entry_epsilon_min (Entry): Entry for epsilon min.
        entry_epsilon_max (Entry): Entry for epsilon max.
        entry_epsilon_decay (Entry): Entry for epsilon decay rate.
        check_decay (Checkbutton): Checkbutton for decaying epsilon.
        decay_checked (bool): Whether the epsilon is decaying.
    """

    def __init__(
            self,
            parent: Misc,
            lbl_w: int,
            default_vals: ExploreParamsDefaults,
            **kwargs
    ):
        """Initialize the epsilon frame.

        Args:
            parent (Misc): The parent of the frame.
            lbl_w (int): The width of the labels.
            default_vals (ExploreParamsDefaults): The default values for
                input fields.
            **kwargs: Keyword arguments for the parent class Label.
        """
        super().__init__(parent, **kwargs)

        self.lbl_epsilon = Label(self, text="Epsilon", width=lbl_w, anchor='e')
        self.lbl_epsilon_min = Label(self, text="Epsilon Min",
                                     width=lbl_w, anchor='e')
        self.lbl_epsilon_max = Label(self, text="Epsilon Max",
                                     width=lbl_w, anchor='e')
        self.lbl_epsilon_decay = Label(self, text="Epsilon Decay Rate",
                                       width=lbl_w, anchor='e')

        self.entry_epsilon = Entry(self)
        self.entry_epsilon.insert(0, default_vals.EPSILON)
        self.entry_epsilon_min = Entry(self)
        self.entry_epsilon_min.insert(0, default_vals.EPSILON_MIN)
        self.entry_epsilon_max = Entry(self)
        self.entry_epsilon_max.insert(0, default_vals.EPSILON_MAX)
        self.entry_epsilon_decay = Entry(self)
        self.entry_epsilon_decay.insert(
            0, default_vals.EPSILON_DECAY_RATE)

        self.check_decay = Checkbutton(
            self,
            text="Decaying Epsilon",
            command=self._on_check_decay
        )
        self.decay_checked = False
        self._on_check_decay()
        self.check_decay.select()

        self.check_decay.grid(row=0, column=0, columnspan=2, sticky='e')
        self.lbl_epsilon.grid(row=1, column=0, sticky='e')
        self.entry_epsilon.grid(row=1, column=1)
        self.lbl_epsilon_min.grid(row=2, column=0, sticky='e')
        self.entry_epsilon_min.grid(row=2, column=1)
        self.lbl_epsilon_max.grid(row=3, column=0, sticky='e')
        self.entry_epsilon_max.grid(row=3, column=1)
        self.lbl_epsilon_decay.grid(row=4, column=0, sticky='e')
        self.entry_epsilon_decay.grid(row=4, column=1)

    def _on_check_decay(self):
        self.decay_checked = not self.decay_checked
        self.update_field_states()

    def update_field_states(self):
        """Update the state of the entry fields."""
        if self.decay_checked:
            self.entry_epsilon.config(state='disabled')
            self.entry_epsilon_min.config(state='normal')
            self.entry_epsilon_max.config(state='normal')
            self.entry_epsilon_decay.config(state='normal')
        else:
            self.entry_epsilon.config(state='normal')
            self.entry_epsilon_min.config(state='disabled')
            self.entry_epsilon_max.config(state='disabled')
            self.entry_epsilon_decay.config(state='disabled')


class HyperparamsFrame(Frame):
    """Frame for setting hyperparameters.

    Attributes:
        lbl_title (Label): Label for the title.
        lbl_alpha (Label): Label for alpha.
        lbl_gamma (Label): Label for gamma.
        entry_alpha (Entry): Entry for alpha.
        entry_gamma (Entry): Entry for gamma.
        epsilon_frame (EpsilonFrame): Frame for epsilon hyperparameters.
    """

    def __init__(
            self,
            parent: Misc,
            lbl_w: int,
            default_vals: ExploreParamsDefaults,
            **kwargs):
        """Initialize the hyperparameters frame.

        Args:
            parent (Misc): The parent of the frame.
            lbl_w (int): The width of the labels.
            default_vals (ExploreParamsDefaults): The default values for
                input fields.
            **kwargs: Keyword arguments for the parent class Label.
        """
        super().__init__(parent, **kwargs)

        # Init labels and entry fields
        self.lbl_title = Label(self, text="Hyperparameters")
        self.lbl_alpha = Label(self, text="Alpha", width=lbl_w, anchor='e')
        self.lbl_gamma = Label(self, text="Gamma", width=lbl_w, anchor='e')
        self.entry_alpha = Entry(self)
        self.entry_alpha.insert(0, default_vals.ALPHA)
        self.entry_gamma = Entry(self)
        self.entry_gamma.insert(0, default_vals.GAMMA)
        # Init epsilon frame
        self.epsilon_frame = EpsilonFrame(self, lbl_w, default_vals, bd=2)

        # Place gui elements in grid
        self.lbl_title.grid(row=0, column=0, columnspan=2)
        self.lbl_alpha.grid(row=1, column=0, sticky='e')
        self.entry_alpha.grid(row=1, column=1)
        self.lbl_gamma.grid(row=2, column=0, sticky='e')
        self.entry_gamma.grid(row=2, column=1)
        self.epsilon_frame.grid(row=3, column=0, columnspan=2, pady=5)

        # Toggle var for fields
        self._epsilon_enabled = True
        self._alpha_enabled = True

    @property
    def alpha(self):
        """Return the alpha value."""
        return float(self.entry_alpha.get())

    @property
    def gamma(self):
        """Return the gamma value."""
        return float(self.entry_gamma.get())

    @property
    def decaying_epsilon(self):
        """Return whether the epsilon is should decay."""
        return self.epsilon_frame.decay_checked

    @property
    def epsilon(self):
        """Return the epsilon value."""
        return float(self.epsilon_frame.entry_epsilon.get())

    @property
    def decaying_epsilon_values(self):
        """Return the epsilon decay preferences."""
        return (
            float(self.epsilon_frame.entry_epsilon_min.get()),
            float(self.epsilon_frame.entry_epsilon_max.get()),
            float(self.epsilon_frame.entry_epsilon_decay.get())
        )

    @property
    def alpha_enabled(self):
        """Return whether alpha fields are enabled."""
        return self._alpha_enabled

    @alpha_enabled.setter
    def alpha_enabled(self, value: bool):
        """Set whether alpha fields are enabled."""
        self._alpha_enabled = value
        state = 'normal' if value else 'disabled'
        self.entry_alpha.config(state=state)
        self.lbl_alpha.config(state=state)

    @property
    def epsilon_enabled(self):
        """Return whether epsilon fields are enabled."""
        return self._epsilon_enabled

    @epsilon_enabled.setter
    def epsilon_enabled(self, value: bool):
        """Set whether epsilon fields are enabled."""
        self._epsilon_enabled = value
        state = 'normal' if value else 'disabled'
        self.epsilon_frame.entry_epsilon.config(state=state)
        self.epsilon_frame.entry_epsilon_min.config(state=state)
        self.epsilon_frame.entry_epsilon_max.config(state=state)
        self.epsilon_frame.entry_epsilon_decay.config(state=state)
        self.epsilon_frame.lbl_epsilon.config(state=state)
        self.epsilon_frame.lbl_epsilon_min.config(state=state)
        self.epsilon_frame.lbl_epsilon_max.config(state=state)
        self.epsilon_frame.lbl_epsilon_decay.config(state=state)
        self.epsilon_frame.check_decay.config(state=state)
        if value:
            self.epsilon_frame.update_field_states()


class ExploreParamsFrame(Frame):
    """Frame for setting exploration parameters.

    Attributes:
        lbl_expl_title (Label): Label for the title.
        lbl_ep_max (Label): Label for max episodes.
        lbl_conv_title (Label): Label for convergence title.
        lbl_conv_rtol (Label): Label for relative tolerance.
        lbl_conv_atol (Label): Label for absolute tolerance.
        entry_conv_rtol (Entry): Entry for relative tolerance.
        entry_conv_atol (Entry): Entry for absolute tolerance.
        entry_ep_max (Entry): Entry for max episodes.
    """

    def __init__(
            self,
            parent: Misc,
            lbl_w: int,
            default_vals: ExploreParamsDefaults,
            **kwargs):
        """Initialize the exploration parameters frame.

        Args:
            parent (Misc): The parent of the frame.
            lbl_w (int): The width of the labels.
            default_vals (ExploreParamsDefaults): The default values for
                input fields.
            **kwargs: Keyword arguments for the parent class Label.
        """
        super().__init__(parent, **kwargs)

        self._conv_tol_enabled = True

        self.lbl_expl_title = Label(self, text="Exploration Parameters")
        self.lbl_ep_max = Label(self, text="Max Episodes (0 = inf.)",
                                width=lbl_w, anchor='e')
        self.lbl_conv_title = Label(self, text="Convergence Tolerance")
        self.lbl_conv_rtol = Label(self, text="Relative Tolerance",
                                   width=lbl_w, anchor='e')
        self.lbl_conv_atol = Label(self, text="Absolute Tolerance",
                                   width=lbl_w, anchor='e')
        self.entry_conv_rtol = Entry(self)
        self.entry_conv_rtol.insert(0, default_vals.CONV_RTOL)
        self.entry_conv_atol = Entry(self)
        self.entry_conv_atol.insert(0, default_vals.CONV_ATOL)
        self.entry_ep_max = Entry(self)
        self.entry_ep_max.insert(0, default_vals.EP_MAX)

        self.lbl_expl_title.grid(row=0, column=0, columnspan=2)
        self.lbl_ep_max.grid(row=1, column=0, sticky='e')
        self.entry_ep_max.grid(row=1, column=1)
        self.lbl_conv_title.grid(row=3, column=0, columnspan=2)
        self.lbl_conv_rtol.grid(row=4, column=0, sticky='e')
        self.entry_conv_rtol.grid(row=4, column=1)
        self.lbl_conv_atol.grid(row=5, column=0, sticky='e')
        self.entry_conv_atol.grid(row=5, column=1)

    @property
    def conv_tol(self):
        """Return the convergence tolerances.

        Returns:
            tuple[float, float]: The relative and absolute tolerance.
        """
        return (
            float(self.entry_conv_rtol.get()),
            float(self.entry_conv_atol.get())
        )

    @property
    def ep_max(self):
        """Return the maximum number of episodes."""
        return int(self.entry_ep_max.get())

    @property
    def conv_tol_enabled(self):
        """Return whether convergence fields are enabled."""
        return self._conv_tol_enabled

    @conv_tol_enabled.setter
    def conv_tol_enabled(self, value: bool):
        """Set whether convergence fields are enabled."""
        self._conv_tol_enabled = value
        state = 'normal' if value else 'disabled'
        self.entry_conv_rtol.config(state=state)
        self.entry_conv_atol.config(state=state)
        self.lbl_conv_rtol.config(state=state)
        self.lbl_conv_atol.config(state=state)
