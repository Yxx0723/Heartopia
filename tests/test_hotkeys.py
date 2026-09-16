import pytest

from actions.hotkeys import HotkeyManager


class FakeWin32Api:
    def __init__(self) -> None:
        self.pressed: set[int] = set()

    def GetAsyncKeyState(self, virtual_key: int) -> int:
        return 0x8000 if virtual_key in self.pressed else 0


def test_hotkey_manager_emits_only_on_press_edge() -> None:
    api = FakeWin32Api()
    manager = HotkeyManager(api=api)

    api.pressed.add(0x75)
    assert manager.poll() == ("screenshot",)
    assert manager.poll() == ()

    api.pressed.clear()
    assert manager.poll() == ()
    api.pressed.add(0x75)
    assert manager.poll() == ("screenshot",)


def test_hotkey_manager_accepts_partial_bindings() -> None:
    api = FakeWin32Api()
    manager = HotkeyManager({"stop": "F10"}, api=api)

    api.pressed.add(0x79)
    assert manager.poll() == ("stop",)
    assert manager.bindings == {"stop": 0x79}


def test_hotkey_manager_rejects_unknown_key() -> None:
    with pytest.raises(ValueError, match="不支持的热键"):
        HotkeyManager({"stop": "ESC"}, api=FakeWin32Api())
