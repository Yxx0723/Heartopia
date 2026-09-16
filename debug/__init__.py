"""调试显示与数据记录模块。"""

from .overlay import DebugOverlay, OverlayError
from .recorder import FailureRecord, FailureRecorder, RecorderError
from .screenshot import ScreenshotError, ScreenshotSaver

__all__ = [
    "DebugOverlay",
    "OverlayError",
    "FailureRecord",
    "FailureRecorder",
    "RecorderError",
    "ScreenshotError",
    "ScreenshotSaver",
]
