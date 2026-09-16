"""视觉检测模块。"""

from .resource_detector import (
    MockResourceDetector,
    MockScenario,
    OpenCVResourceDetector,
    ResourceDetector,
)
from .prompt_detector import PromptDetection, PromptDetector, PromptDetectorError
from .motion_detector import MotionDetector, MotionDetectorError, MotionResult
from .target_selector import SelectionWeights, TargetScore, TargetSelector
from .ui_detector import UiDetector, UiDetectorError, UiState

__all__ = [
    "MockResourceDetector",
    "MockScenario",
    "OpenCVResourceDetector",
    "PromptDetection",
    "PromptDetector",
    "PromptDetectorError",
    "MotionDetector",
    "MotionDetectorError",
    "MotionResult",
    "ResourceDetector",
    "SelectionWeights",
    "TargetScore",
    "TargetSelector",
    "UiDetector",
    "UiDetectorError",
    "UiState",
]
