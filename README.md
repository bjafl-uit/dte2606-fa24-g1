# DTE2606 Fall 2024 - Reinforcement learning

## Getting started

- App can be launched by running [\src\main.py](src/main.py)
- Default config in [src/default_config](src/default_config.py)
- Config file may be edited, but parameters can be set at runtime

## Modules

- The most interesting module is [agent](src/agent/), which contains implementation of the RL algorithms
- The module [environment](src/environment/) houses the environment model
- Other models are mainly supporting modules for [GUI](src/gui), [pygame simulation](src/simulation/) and [config classes](src/config/)

- Report can be found in [docs folder](/docs/report/dte2606-fa24-graded1-bfl.pdf)

## Requirements

- matplotlib
- numpy
- pygame

Listed in [requirements.txt](requirements.txt). May be installed with `pip install -r requirements.txt`

## Notes

- AI tools (ChatGPT and GH Copilot) have been applied for support, like code completion, productivity and support in syntax lookup and providing examples.
