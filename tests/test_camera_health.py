from uniscan.ui.camera_health import camera_health_state


def test_camera_health_closed() -> None:
    state = camera_health_state(is_open=False, is_previewing=False)
    assert state.label == "Camera: Closed"


def test_camera_health_open() -> None:
    state = camera_health_state(is_open=True, is_previewing=False)
    assert state.label == "Camera: Open"


def test_camera_health_previewing() -> None:
    state = camera_health_state(is_open=True, is_previewing=True)
    assert state.label == "Camera: Previewing"


def test_camera_health_error_overrides_other_states() -> None:
    state = camera_health_state(is_open=True, is_previewing=True, error_text="fail")
    assert state.label == "Camera: Error"


def test_camera_health_opening_state() -> None:
    state = camera_health_state(is_open=False, is_previewing=False, is_opening=True)
    assert state.label == "Camera: Opening..."


def test_camera_health_error_overrides_opening() -> None:
    state = camera_health_state(
        is_open=False, is_previewing=False, is_opening=True, error_text="fail"
    )
    assert state.label == "Camera: Error"


def test_camera_health_detail_is_appended_when_open_or_previewing() -> None:
    previewing = camera_health_state(is_open=True, is_previewing=True, detail="1920x1080 @ 28 fps")
    assert previewing.label == "Camera: Previewing (1920x1080 @ 28 fps)"

    opened = camera_health_state(is_open=True, is_previewing=False, detail="1920x1080")
    assert opened.label == "Camera: Open (1920x1080)"

    closed = camera_health_state(is_open=False, is_previewing=False, detail="1920x1080")
    assert closed.label == "Camera: Closed"
