"""Trace file writer for crash-safe, async-safe JSONL trace logging.

Provides TraceWriter for writing TraceEntry objects to JSONL files with
flush+fsync for crash safety and asyncio.Lock + asyncio.to_thread for
non-blocking, race-safe writes. Also provides _resolve_trace_writer to
create or propagate a TraceWriter from run() kwargs.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from quanted_agents.observability import TraceEntry


class TraceWriter:
    """Writes TraceEntry objects to a JSONL file with async-safe, crash-safe writes.

    Uses asyncio.Lock to serialize concurrent coroutine writes and
    asyncio.to_thread() to offload blocking file I/O off the event loop.

    Each write opens, appends, flushes, fsyncs, and closes the file.
    This ensures crash safety and avoids dangling file handles.
    """

    def __init__(self, file_path: Path) -> None:
        """Create a TraceWriter for the given file path.

        Args:
            file_path: Path to the JSONL file to write. The parent
                directory must already exist.
        """
        self._file_path: Path = file_path
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def file_path(self) -> Path:
        """The path to the trace file being written.

        Returns:
            The Path object for the JSONL trace file.
        """
        return self._file_path

    async def write(self, entry: TraceEntry) -> None:
        """Write a single trace entry as a JSONL line.

        Serializes the entry to JSON, acquires the async lock,
        then writes via asyncio.to_thread() for non-blocking I/O.

        Args:
            entry: The TraceEntry to serialize and append.
        """
        line = json.dumps(entry.to_dict()) + "\n"
        async with self._lock:
            await asyncio.to_thread(self._write_sync, line)

    def _write_sync(self, line: str) -> None:
        """Synchronous write with flush and fsync for crash safety.

        Opens the file in append mode, writes the line, flushes
        Python buffers, and calls os.fsync() to ensure durability.

        Args:
            line: The pre-serialized JSONL line to write.
        """
        with open(self._file_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


class TraceSession:
    """Context manager that consolidates multiple agent runs into a single trace file.

    All agent.run() calls within the session block that pass trace_session=session
    write their trace entries to the same JSONL file. Generates a session_id (UUID)
    attached to every trace entry for correlation.

    On context exit (even on exception), the session remains valid -- TraceWriter
    writes are flushed per-entry, so partial traces are preserved.

    Args:
        file_path: Path to the JSONL file for consolidated traces.
    """

    def __init__(self, file_path: str | Path) -> None:
        """Create a TraceSession for the given file path.

        Args:
            file_path: Path to the JSONL file for consolidated traces.
        """
        self._file_path: Path = Path(file_path)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer: TraceWriter = TraceWriter(self._file_path)
        self._session_id: str = str(uuid.uuid4())

    @property
    def session_id(self) -> str:
        """The unique session identifier (UUID).

        Returns:
            A UUID string identifying this session.
        """
        return self._session_id

    @property
    def writer(self) -> TraceWriter:
        """The TraceWriter used for this session.

        Returns:
            The TraceWriter instance writing to the session file.
        """
        return self._writer

    @property
    def file_path(self) -> Path:
        """The path to the session trace file.

        Returns:
            The Path object for the JSONL trace file.
        """
        return self._file_path

    async def __aenter__(self) -> TraceSession:
        """Enter async context manager.

        Returns:
            This TraceSession instance.
        """
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Exit async context manager.

        No-op: TraceWriter writes are already flushed per-entry.
        Session valid on exception (partial traces > no traces).

        Args:
            exc_type: The exception type, if an exception was raised.
            exc_val: The exception value, if an exception was raised.
            exc_tb: The traceback, if an exception was raised.
        """
        pass


def _resolve_trace_writer(kwargs: dict[str, Any]) -> TraceWriter | None:
    """Create or propagate a TraceWriter from run() kwargs.

    Pops traces_path, trace_filename, trace_session, and _trace_writer from
    kwargs to prevent them from propagating to child steps or pydantic-ai's
    Agent.run(). Priority: trace_session > _trace_writer > traces_path.

    When traces_path is set and trace_filename is provided, the trace file
    uses that custom name. Without trace_filename, an auto-generated
    timestamped name is used. trace_filename without traces_path is ignored.

    Args:
        kwargs: The **kwargs dict from a run() method. Modified in-place
            to remove trace-related keys.

    Returns:
        A TraceWriter instance if a trace destination was configured,
        or None if no trace configuration was provided.
    """
    traces_path = kwargs.pop("traces_path", None)
    trace_filename = kwargs.pop("trace_filename", None)
    trace_session = kwargs.pop("trace_session", None)
    trace_writer = kwargs.pop("_trace_writer", None)

    # trace_session takes highest precedence
    if trace_session is not None:
        return trace_session.writer

    if trace_writer is not None:
        return trace_writer

    if traces_path is not None:
        path = Path(traces_path)
        path.mkdir(parents=True, exist_ok=True)
        if trace_filename is not None:
            return TraceWriter(path / trace_filename)
        filename = f"trace_{datetime.now().strftime('%Y%m%dT%H%M%S_%f')}.jsonl"
        return TraceWriter(path / filename)

    return None
