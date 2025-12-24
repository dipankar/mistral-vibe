"""Session lock to prevent multiple simultaneous vibe sessions."""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from vibe.core.config_path import SESSION_LOCK_FILE

logger = logging.getLogger(__name__)

# Track if we've registered the atexit handler
_atexit_registered = False
_lock_fd: int | None = None


class SessionLockError(Exception):
    """Raised when another session is already active."""

    def __init__(self, pid: int, start_time: str) -> None:
        self.pid = pid
        self.start_time = start_time
        super().__init__(
            f"Another vibe session is already running (PID: {pid}, started: {start_time})"
        )


def _is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _read_lock_file() -> tuple[int, str] | None:
    """Read lock file and return (pid, start_time) or None if invalid."""
    lock_path = SESSION_LOCK_FILE.path
    if not lock_path.exists():
        return None

    try:
        content = lock_path.read_text().strip()
        lines = content.split("\n")
        if len(lines) >= 2:
            pid = int(lines[0])
            start_time = lines[1]
            return pid, start_time
    except (ValueError, OSError) as e:
        logger.debug("Failed to read lock file: %s", e)
    return None


def _write_lock_content(fd: int) -> None:
    """Write current process info to lock file descriptor."""
    pid = os.getpid()
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    content = f"{pid}\n{start_time}\n"

    # Truncate and write
    os.ftruncate(fd, 0)
    os.lseek(fd, 0, os.SEEK_SET)
    os.write(fd, content.encode())


def _remove_lock_file() -> None:
    """Remove the lock file and release file lock."""
    global _lock_fd

    try:
        lock_path = SESSION_LOCK_FILE.path

        # Release file lock if we hold it
        if _lock_fd is not None:
            try:
                fcntl.flock(_lock_fd, fcntl.LOCK_UN)
                os.close(_lock_fd)
            except OSError as e:
                logger.debug("Failed to release file lock: %s", e)
            _lock_fd = None

        # Only remove if we own the lock
        if lock_path.exists():
            lock_info = _read_lock_file()
            if lock_info and lock_info[0] == os.getpid():
                lock_path.unlink()
                logger.debug("Session lock released")
    except OSError as e:
        logger.debug("Failed to remove lock file: %s", e)


def is_session_active() -> tuple[bool, int | None, str | None]:
    """Check if another session is currently active.

    Returns:
        Tuple of (is_active, pid, start_time)
    """
    lock_info = _read_lock_file()
    if not lock_info:
        return False, None, None

    pid, start_time = lock_info

    # Check if it's our own process
    if pid == os.getpid():
        return False, None, None

    # Check if the process is still running
    if _is_process_running(pid):
        return True, pid, start_time

    # Stale lock file - process no longer exists
    logger.debug("Found stale lock file from PID %d", pid)
    return False, None, None


def acquire_session_lock() -> bool:
    """Acquire the session lock atomically using flock.

    Returns:
        True if lock acquired, False if another session is active.
    """
    global _lock_fd, _atexit_registered

    lock_path = SESSION_LOCK_FILE.path
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Open or create the lock file
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)

        # Try to acquire exclusive lock (non-blocking)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            # Another process holds the lock
            os.close(fd)

            # Read lock info for error message
            lock_info = _read_lock_file()
            if lock_info:
                logger.debug(
                    "Lock held by PID %d (started %s)",
                    lock_info[0], lock_info[1]
                )
            return False

        # Check if existing lock is from a dead process
        existing = _read_lock_file()
        if existing and existing[0] != os.getpid():
            if _is_process_running(existing[0]):
                # Process is still running but we got the flock?
                # This shouldn't happen, but be safe
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)
                return False

        # We have the lock - write our info
        _write_lock_content(fd)
        _lock_fd = fd

        # Register cleanup handler
        if not _atexit_registered:
            atexit.register(_remove_lock_file)
            _atexit_registered = True

        logger.debug("Session lock acquired (PID %d)", os.getpid())
        return True

    except OSError as e:
        logger.warning("Failed to acquire session lock: %s", e)
        return False


def release_session_lock() -> None:
    """Release the session lock."""
    _remove_lock_file()


@contextmanager
def session_lock() -> Generator[None, None, None]:
    """Context manager for session lock."""
    if not acquire_session_lock():
        lock_info = _read_lock_file()
        if lock_info:
            raise SessionLockError(lock_info[0], lock_info[1])
        raise SessionLockError(0, "unknown")

    try:
        yield
    finally:
        release_session_lock()
