from __future__ import annotations

import queue
import threading

from uniscan.ui.app import UnifiedScanApp


class _Var:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class _Widget:
    def __init__(self) -> None:
        self.options = {"state": "disabled", "text": ""}

    def configure(self, **kwargs) -> None:
        self.options.update(kwargs)

    def cget(self, name):
        return self.options[name]


class _Progress:
    def __init__(self) -> None:
        self.mode = "determinate"
        self.value = 0.0
        self.running = False

    def configure(self, *, mode) -> None:
        self.mode = mode

    def set(self, value) -> None:
        self.value = float(value)

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False


def _app() -> UnifiedScanApp:
    app = object.__new__(UnifiedScanApp)
    app.job_queue = queue.Queue()
    app.job_cancel_event = threading.Event()
    app.job_thread = None
    app._job_state = "idle"
    app._job_name = None
    app._job_cancel_supported = False
    app._job_cancel_before_commit = False
    app._job_progress_value = None
    app._job_generation = 0
    app._job_retry_callback = None
    app._closing = False
    app._pending_stage_history_notice = None
    app.status_var = _Var()
    app.cancel_task_button = _Widget()
    app.retry_task_button = _Widget()
    app.job_progress_bar = _Progress()
    app.winfo_exists = lambda: False
    return app


def _join(app: UnifiedScanApp) -> None:
    thread = app.job_thread
    assert thread is not None
    thread.join(timeout=3)
    assert not thread.is_alive()


def test_job_display_switches_between_indeterminate_and_determinate() -> None:
    app = _app()

    app._set_job_display(stage="Import", current="page 3", progress=None)
    assert app.job_progress_bar.mode == "indeterminate"
    assert app.job_progress_bar.running is True
    assert app.status_var.get() == "Import | page 3"

    app._set_job_display(stage="Import", current="page 3/10", progress=45)
    assert app.job_progress_bar.mode == "determinate"
    assert app.job_progress_bar.running is False
    assert app.job_progress_bar.value == 0.45
    assert app.status_var.get().endswith("45%")


def test_job_success_and_stale_generation_events(monkeypatch) -> None:
    app = _app()
    completed = []
    monkeypatch.setattr("uniscan.ui.app.messagebox.showwarning", lambda *_args: None)

    assert app._start_background_job(
        "Import",
        lambda emit, _cancelled: (emit(stage="Import", current="1/1", progress=100), 7)[1],
        completed.append,
    )
    _join(app)
    app._poll_job_queue()

    assert completed == [7]
    assert app._job_state == "success"
    assert app.job_thread is None
    assert app.job_progress_bar.value == 1.0
    assert app.cancel_task_button.cget("state") == "disabled"

    active_thread = object()
    app._job_generation = 2
    app._job_state = "running"
    app.job_thread = active_thread
    app.status_var.set("new job")
    app.job_queue.put(("progress", (1, "Old", "stale", 99)))
    app.job_queue.put(("done", (1, completed.append, 9, "Old", None)))
    app._poll_job_queue()
    assert app.status_var.get() == "new job"
    assert app.job_thread is active_thread
    assert completed == [7]


def test_cancel_after_worker_before_commit_is_idempotent_and_preserves_state() -> None:
    app = _app()
    committed = []
    cleaned = []
    retried = []

    class FinishedThread:
        @staticmethod
        def is_alive() -> bool:
            return False

    app.job_thread = FinishedThread()
    app._job_generation = 1
    app._job_state = "running"
    app._job_name = "Apply preview"
    app._job_cancel_supported = True
    app._job_cancel_before_commit = True
    app._job_retry_callback = lambda: retried.append(True)

    app.cancel_current_job()
    assert app.job_cancel_event.is_set()
    assert app._job_state == "cancelling"
    assert app.cancel_task_button.cget("state") == "disabled"
    app.cancel_current_job()
    assert app.status_var.get() == "Cancellation already requested."

    app.job_queue.put(
        (
            "done",
            (1, committed.append, "candidate", "Apply preview", lambda: cleaned.append(True)),
        )
    )
    app._poll_job_queue()
    assert committed == []
    assert cleaned == [True]
    assert app._job_state == "cancelled"
    assert app.job_thread is None
    assert "previous committed pages were preserved" in app.status_var.get()
    assert app.retry_task_button.cget("state") == "normal"


def test_error_releases_busy_slot_and_retry_starts_fresh_job(monkeypatch) -> None:
    app = _app()
    errors = []
    recovered = []
    monkeypatch.setattr(
        "uniscan.ui.app.messagebox.showerror",
        lambda title, text: errors.append((title, text)),
    )

    def retry() -> None:
        app._start_background_job(
            "Recovered",
            lambda _emit, _cancelled: "ok",
            recovered.append,
        )

    assert app._start_background_job(
        "Apply preview",
        lambda _emit, _cancelled: (_ for _ in ()).throw(RuntimeError("processor failed")),
        lambda _result: None,
        retry=retry,
    )
    _join(app)
    app._poll_job_queue()
    assert app._job_state == "error"
    assert app.job_thread is None
    assert app.retry_task_button.cget("state") == "normal"
    assert "previous committed pages were preserved" in app.status_var.get()
    assert errors == [("Apply preview Error", "processor failed")]

    app.retry_last_job()
    _join(app)
    app._poll_job_queue()
    assert recovered == ["ok"]
    assert app._job_state == "success"
    assert app.retry_task_button.cget("state") == "disabled"


def test_busy_rejection_and_incomplete_rollback_wording(monkeypatch) -> None:
    app = _app()
    warnings = []
    app.job_thread = object()
    monkeypatch.setattr(
        "uniscan.ui.app.messagebox.showwarning",
        lambda title, text: warnings.append((title, text)),
    )

    assert not app._start_background_job(
        "Second",
        lambda _emit, _cancelled: None,
        lambda _result: None,
    )
    assert warnings == [("Busy", "Another background job is already running.")]
    assert app._job_generation == 0
    assert (
        app._job_failure_status("Apply", "rollback was incomplete")
        == "Apply failed: rollback incomplete; keep the app open and retry or recover."
    )


def test_post_commit_failure_does_not_claim_pages_were_preserved(monkeypatch) -> None:
    app = _app()
    errors = []
    monkeypatch.setattr(
        "uniscan.ui.app.messagebox.showerror",
        lambda title, text: errors.append((title, text)),
    )

    def on_done(_result) -> None:
        app._job_commit_completed = True
        raise RuntimeError("refresh failed")

    assert app._start_background_job(
        "Apply preview",
        lambda _emit, _cancelled: "candidate",
        on_done,
    )
    _join(app)
    app._poll_job_queue()

    assert app._job_state == "error"
    assert "committed changes" in app.status_var.get()
    assert "previous committed pages were preserved" not in app.status_var.get()
    assert app._job_retry_callback is None
    assert app.retry_task_button.cget("state") == "disabled"
    assert errors == [("Apply preview Error", "refresh failed")]


def test_closing_discards_staged_result_without_running_done_callback() -> None:
    app = _app()
    app._closing = True
    committed = []
    cleaned = []
    app.job_queue.put(
        (
            "done",
            (1, committed.append, "candidate", "Apply preview", lambda: cleaned.append(True)),
        )
    )

    app._poll_job_queue()

    assert committed == []
    assert cleaned == [True]
