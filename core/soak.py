"""离线状态机压测和故障注入工具。

该模块使用 Perception 序列模拟目标出现、交互成功和异常路径，只验证决策
层不会死锁、不会把状态推进到 STOPPED，也不会替代真实游戏窗口压测。
"""

from __future__ import annotations

from dataclasses import dataclass

from core.models import Detection, Perception
from core.state import BotState
from core.state_machine import StateMachine


@dataclass(frozen=True)
class SoakReport:
    """一次离线压测的可读结果。"""

    requested_targets: int
    collected_targets: int
    ticks: int
    failures: tuple[str, ...]
    recovered_faults: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return (
            self.requested_targets == self.collected_targets
            and not self.failures
            and len(self.recovered_faults) == 3
        )


def run_offline_soak(target_count: int = 20) -> SoakReport:
    """运行目标采集主路径和三类故障注入。"""
    if target_count <= 0:
        raise ValueError("target_count 必须大于零")

    machine = StateMachine(lost_target_frames=2)
    failures: list[str] = []
    ticks = 0
    now = 0.0

    def tick(perception: Perception) -> None:
        nonlocal ticks, now
        action = machine.update(perception, now)
        ticks += 1
        now += 0.1
        if action.kind == "stop" and machine.state is BotState.STOPPED:
            failures.append(f"unexpected_stop_at_tick_{ticks}")

    tick(_perception())
    for index in range(target_count):
        target = Detection(
            "wood",
            850 + (index % 5) * 15,
            460,
            930 + (index % 5) * 15,
            540,
            0.90,
        )
        tick(_perception([target]))
        tick(_perception([target]))
        tick(_perception([target]))
        tick(_perception([target], prompt=True))
        tick(_perception([target], prompt=True))
        tick(_perception())

    recovered_faults = _run_fault_checks()
    if len(recovered_faults) != 3:
        failures.append("fault_injection")
    if machine.state is not BotState.PATROL:
        failures.append(f"final_state_{machine.state.name}")

    return SoakReport(
        requested_targets=target_count,
        collected_targets=machine.context.total_collected,
        ticks=ticks,
        failures=tuple(failures),
        recovered_faults=recovered_faults,
    )


def _run_fault_checks() -> tuple[str, ...]:
    recovered: list[str] = []

    lost_machine = StateMachine(lost_target_frames=2)
    _enter_approach(lost_machine)
    lost_machine.update(_perception(), 1.0)
    lost_machine.update(_perception(), 1.1)
    if lost_machine.state is BotState.PATROL:
        recovered.append("target_lost")

    recovery_machine = StateMachine(target_timeout_seconds=0.5)
    _enter_approach(recovery_machine)
    recovery_machine.update(_perception([_target()]), 1.0)
    if recovery_machine.state is BotState.RECOVERY:
        recovery_machine.complete_recovery(1.1)
        if recovery_machine.state is BotState.PATROL:
            recovered.append("recovery")

    ui_machine = StateMachine()
    ui_machine.update(_perception(), 0.0)
    ui_machine.update(Perception([], False, 0.0, abnormal_ui=True), 0.1)
    if ui_machine.state is BotState.PAUSED:
        recovered.append("blocking_ui")

    return tuple(recovered)


def _enter_approach(machine: StateMachine) -> None:
    machine.update(_perception(), 0.0)
    target = _target()
    machine.update(_perception([target]), 0.1)
    machine.update(_perception([target]), 0.2)
    machine.update(_perception([target]), 0.3)


def _target() -> Detection:
    return Detection("wood", 850, 460, 930, 540, 0.9)


def _perception(
    resources: list[Detection] | None = None,
    *,
    prompt: bool = False,
) -> Perception:
    return Perception(
        resources=list(resources or []),
        interaction_prompt=prompt,
        motion_score=0.5,
        frame_size=(1920, 1080),
    )


__all__ = ["SoakReport", "run_offline_soak"]
