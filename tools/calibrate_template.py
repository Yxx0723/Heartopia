"""从静态截图裁剪 OpenCV 模板。

示例：
    python -m tools.calibrate_template --input screenshot.png --output assets/resources/wood.png
    python -m tools.calibrate_template --input screenshot.png --output wood.png --bbox 100 200 180 300
"""

from __future__ import annotations

import argparse

import cv2

from vision.template_tools import TemplateCalibrator, TemplateError


def main() -> int:
    parser = argparse.ArgumentParser(description="从静态截图制作资源/交互模板")
    parser.add_argument("--input", required=True, help="输入截图路径")
    parser.add_argument("--output", required=True, help="输出模板路径")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="可选的像素框；省略时打开 ROI 框选窗口",
    )
    parser.add_argument("--margin", type=int, default=0, help="裁剪边缘扩展像素")
    args = parser.parse_args()

    try:
        calibrator = TemplateCalibrator(cv2_module=cv2)
        if args.bbox:
            image = cv2.imread(args.input, cv2.IMREAD_COLOR)
            if image is None:
                raise TemplateError(f"无法读取输入截图: {args.input}")
            info = calibrator.save_crop(
                image,
                tuple(args.bbox),
                args.output,
                margin=args.margin,
            )
        else:
            info = calibrator.select_and_save(args.input, args.output)
    except (OSError, TemplateError, ValueError) as exc:
        print(f"模板处理失败: {exc}")
        return 1

    print(
        f"模板已保存: {info.path} "
        f"size={info.width}x{info.height} channels={info.channels} "
        f"std={info.standard_deviation:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
