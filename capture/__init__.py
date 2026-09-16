"""游戏窗口查找和截图模块。"""

from .capture import CaptureError, WindowCapture
from .window import GameWindow, WindowError, WindowNotFoundError

__all__ = [
    "CaptureError",
    "GameWindow",
    "WindowCapture",
    "WindowError",
    "WindowNotFoundError",
]
