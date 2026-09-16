from pathlib import Path

import cv2
import numpy as np
import pytest

from vision.template_tools import TemplateCalibrator, TemplateError, inspect_template


def test_crop_and_save_template(tmp_path: Path) -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[5:15, 8:18] = (20, 80, 180)
    frame[8:12, 10:14] = (180, 40, 20)
    output = tmp_path / "wood.png"

    info = TemplateCalibrator().save_crop(frame, (8, 5, 18, 15), output)

    assert info.path == output.resolve()
    assert (info.width, info.height, info.channels) == (10, 10, 3)
    assert output.is_file()
    assert inspect_template(output).standard_deviation > 0


def test_crop_clips_margin_to_frame() -> None:
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[0:5, 0:5] = (10, 30, 100)

    result = TemplateCalibrator().crop(frame, (0, 0, 10, 10), margin=20)

    assert result.shape == (20, 30, 3)


def test_constant_template_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "constant.png"
    cv2.imwrite(str(path), np.full((10, 10), 128, dtype=np.uint8))

    with pytest.raises(TemplateError, match="像素变化过小"):
        inspect_template(path)


def test_invalid_bbox_is_rejected() -> None:
    with pytest.raises(TemplateError, match="有效矩形"):
        TemplateCalibrator().crop(
            np.zeros((10, 10, 3), dtype=np.uint8),
            (8, 8, 2, 2),
        )
