"""键鼠输入和快捷键控制模块。"""

from .controller import InputController, InputError
from .hotkeys import HotkeyError, HotkeyManager

__all__ = ["InputController", "InputError", "HotkeyError", "HotkeyManager"]
