from __future__ import annotations

from pathlib import Path

from vibe.core import session_lock
from vibe.core.config_path import ConfigPath


def test_session_lock_with_null_backend(monkeypatch, tmp_path: Path) -> None:
    fake_backend = session_lock._NullLockBackend()
    monkeypatch.setattr(session_lock, "_LOCK_BACKEND", fake_backend, raising=False)
    monkeypatch.setattr(session_lock, "_lock_fd", None, raising=False)

    fake_lock = tmp_path / "session.lock"
    monkeypatch.setattr(
        session_lock,
        "SESSION_LOCK_FILE",
        ConfigPath(lambda: fake_lock),
        raising=False,
    )

    try:
        assert session_lock.acquire_session_lock() is True
        assert fake_lock.exists()
    finally:
        session_lock.release_session_lock()

    assert not fake_lock.exists()
