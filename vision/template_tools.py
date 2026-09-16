"""资源和交互模板的离线检查、裁剪与校准工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class TemplateError(ValueError):
    """模板读取、裁剪或质量校验失败时抛出。"""


@dataclass(frozen=True)
class TemplateInfo:
    """模板图像的基础质量信息。"""

    path: Path | None
    width: int
    height: int
    channels: int
    mean: float
    standard_deviation: float

    @property
    def is_constant(self) -> bool:
        """返回模板是否几乎没有可匹配的像素变化。"""
        return self.standard_deviation <= 1e-6


class TemplateCalibrator:
    """提供可测试的模板裁剪保存能力，以及可选的 OpenCV ROI 选择。"""

    def __init__(self, *, cv2_module: Any = cv2) -> None:
        self._cv2 = cv2_module

    def inspect(self, path: str | Path) -> TemplateInfo:
        """读取并检查一个模板文件。"""
        template_path = Path(path)
        try:
            image = self._cv2.imread(str(template_path), self._cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            raise TemplateError(f"读取模板失败: {template_path}") from exc
        if image is None:
            raise TemplateError(f"无法读取模板: {template_path}")
        info = _image_info(image, template_path, self._cv2)
        if info.width < 4 or info.height < 4:
            raise TemplateError("模板尺寸至少需要 4x4 像素")
        if info.is_constant:
            raise TemplateError("模板像素变化过小，无法进行可靠匹配")
        return info

    def crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        margin: int = 0,
    ) -> np.ndarray:
        """按客户区像素坐标裁剪模板，margin 会被限制在图像边界内。"""
        _validate_frame(frame)
        if len(bbox) != 4:
            raise TemplateError("bbox 必须包含 x1, y1, x2, y2")
        x1, y1, x2, y2 = (int(value) for value in bbox)
        if margin < 0:
            raise TemplateError("margin 不能为负数")
        height, width = frame.shape[:2]
        x1 = max(0, x1 - margin)
        y1 = max(0, y1 - margin)
        x2 = min(width, x2 + margin)
        y2 = min(height, y2 + margin)
        if x1 >= x2 or y1 >= y2:
            raise TemplateError("bbox 必须是图像内的有效矩形")
        return np.ascontiguousarray(frame[y1:y2, x1:x2].copy())

    def save_crop(
        self,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        output_path: str | Path,
        *,
        margin: int = 0,
    ) -> TemplateInfo:
        """裁剪并保存模板，然后重新检查保存后的文件。"""
        crop = self.crop(frame, bbox, margin=margin)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = self._cv2.imwrite(str(path), crop)
        except Exception as exc:
            raise TemplateError(f"保存模板失败: {path}") from exc
        if not written:
            raise TemplateError(f"模板编码失败: {path}")
        return self.inspect(path)

    def select_and_save(
        self,
        image_path: str | Path,
        output_path: str | Path,
        *,
        window_name: str = "Select template ROI",
    ) -> TemplateInfo:
        """打开静态图片，让用户框选 ROI 后保存模板。"""
        source_path = Path(image_path)
        image = self._cv2.imread(str(source_path), self._cv2.IMREAD_COLOR)
        if image is None:
            raise TemplateError(f"无法读取校准图片: {source_path}")
        try:
            x, y, width, height = self._cv2.selectROI(
                window_name,
                image,
                showCrosshair=True,
                fromCenter=False,
            )
        finally:
            destroy = getattr(self._cv2, "destroyWindow", None)
            if destroy is not None:
                destroy(window_name)
        if width <= 0 or height <= 0:
            raise TemplateError("未选择有效 ROI")
        return self.save_crop(
            image,
            (int(x), int(y), int(x + width), int(y + height)),
            output_path,
        )


def inspect_template(path: str | Path) -> TemplateInfo:
    """使用 OpenCV 快捷检查模板文件。"""
    return TemplateCalibrator().inspect(path)


def _image_info(image: np.ndarray, path: Path | None, cv2_module: Any) -> TemplateInfo:
    if image.ndim == 2:
        gray = image
        channels = 1
    elif image.ndim == 3:
        channels = int(image.shape[2])
        conversion = {
            3: getattr(cv2_module, "COLOR_BGR2GRAY", 6),
            4: getattr(cv2_module, "COLOR_BGRA2GRAY", 10),
        }.get(channels)
        if conversion is None:
            raise TemplateError("模板必须是灰度、BGR 或 BGRA 图像")
        gray = cv2_module.cvtColor(image, conversion)
    else:
        raise TemplateError("模板图像维度不受支持")
    return TemplateInfo(
        path=path.resolve() if path is not None else None,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        channels=channels,
        mean=float(np.mean(gray)),
        standard_deviation=float(np.std(gray)),
    )


def _validate_frame(frame: np.ndarray) -> None:
    if not isinstance(frame, np.ndarray):
        raise TemplateError("frame 必须是 numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] < 3:
        raise TemplateError("frame 必须是 H x W x 3 的 BGR 图像")


__all__ = ["TemplateCalibrator", "TemplateError", "TemplateInfo", "inspect_template"]
