"""GUI: HTML/JS panel for document viewer and agent log (e.g. in Jupyter/Colab)."""

from .panel import (
    GUILoggerAdapter,
    GUIInterface,
    create_gui_panel_v12,
    run_with_gui,
)

__all__ = [
    "create_gui_panel_v12",
    "GUIInterface",
    "GUILoggerAdapter",
    "run_with_gui",
]
