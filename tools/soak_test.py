"""运行离线 20 目标压测，不连接游戏窗口。"""

from __future__ import annotations

import argparse

from core.soak import run_offline_soak


def main() -> int:
    parser = argparse.ArgumentParser(description="运行状态机离线目标压测")
    parser.add_argument("--targets", type=int, default=20)
    args = parser.parse_args()
    try:
        report = run_offline_soak(args.targets)
    except ValueError as exc:
        print(f"压测参数错误: {exc}")
        return 2

    print(
        f"requested={report.requested_targets} "
        f"collected={report.collected_targets} ticks={report.ticks}"
    )
    print(f"recovered_faults={','.join(report.recovered_faults) or '-'}")
    if report.failures:
        print(f"failures={','.join(report.failures)}")
        return 1
    print("OFFLINE SOAK PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
