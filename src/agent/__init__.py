"""Module for implementation of the agent and learning models.

This module contains classes for the robot agent, it's learning models and
episode history. The module plot contains plotting functions to plot
exploration data.

Classes:
    Robot: The robot agent.
    LearningModel: The learning model.
    MonteCarlo: The Monte Carlo policy
    QLearning: The Q-learning policy.
    DynamicEpsilon: The dynamically decaying epsilon.
    Episode: The episode history.

Functions:
    plot_exploration_data: Plotting function for exploration data.

"""
from .robot import Robot
from .learning_model import DynamicEpsilon, LearningModel
from .q_learning import QLearning
from .monte_carlo import MonteCarlo

from .episode import Episode
from .plot import plot_exploration_data

__all__ = ['Robot', 'DynamicEpsilon', 'LearningModel', 'QLearning',
           'MonteCarlo', 'Episode', 'plot_exploration_data']
