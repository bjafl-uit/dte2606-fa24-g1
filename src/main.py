"""Main entry point for the application."""

from gui import MainMenu
from default_config import GRID_WORLD_PREFS, EXPLORE_PARAMS_DEFAULT

if __name__ == '__main__':
    main_window = MainMenu(GRID_WORLD_PREFS, EXPLORE_PARAMS_DEFAULT)
    main_window.mainloop()
