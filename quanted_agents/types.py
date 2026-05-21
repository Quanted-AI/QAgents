"""Type aliases and protocols for the quanted_agents package."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Union, runtime_checkable

from pydantic import BaseModel

if TYPE_CHECKING:
    from quanted_agents.artifact_store import ArtifactStore
    from quanted_agents.result import QuantedResult
    from quanted_agents.workflows.parallel import ParallelResult

_T = TypeVar("_T")

AssemblyFn = Union[
    Callable[["ArtifactStore", "QuantedResult"], _T],
    Callable[["ArtifactStore", "QuantedResult"], Awaitable[_T]],
]
"""Type alias for assembly functions used by Pipeline and Loop.

An assembly function receives the ArtifactStore and the last QuantedResult,
and returns a transformed output. May be sync or async.
"""

ParallelAssemblyFn = Union[
    Callable[["ArtifactStore", "ParallelResult"], _T],
    Callable[["ArtifactStore", "ParallelResult"], Awaitable[_T]],
]
"""Type alias for assembly functions used by Parallel.

Same as AssemblyFn but the second argument is ParallelResult, giving
access to .results and .errors properties.
"""

InputTransformFn = Union[
    Callable[["ArtifactStore", str], Any],
    Callable[["ArtifactStore", str], Awaitable[Any]],
]
"""Transforms the parent LLM's instruction into the child Runnable's input.

Receives the ArtifactStore at invocation time (NOT captured at registration)
and the parent LLM's instruction string. Returns the input for the child
Runnable -- typically a BaseModel instance.

May be sync or async. When async, RunnableTool awaits it before calling
the child Runnable.
"""

PipelineTransformFn = Union[
    Callable[["QuantedResult", "ArtifactStore", int], BaseModel],
    Callable[["QuantedResult", "ArtifactStore", int], Awaitable[BaseModel]],
]
"""Transform function for bridging type gaps between Pipeline stages.

Receives the previous stage's full QuantedResult, the Pipeline's ArtifactStore,
and the current stage index. Returns a BaseModel instance suitable as input
for the current stage.

Distinct from InputTransformFn, which is used by RunnableTool to transform
a parent LLM's instruction string into child input. PipelineTransformFn
operates at stage boundaries within a Pipeline workflow.

May be sync or async.
"""

ToolType = Any
"""Type alias for tool functions passed to pydantic-ai.

pydantic-ai's internal tool type is complex and varies by registration method.
Using Any is the practical approach for the public API.
"""


class OverflowStrategy(str, Enum):
    """Strategy for handling context window overflow.

    Determines behavior when estimated token count exceeds max_context_tokens.
    """

    RAISE = "raise"
    TRUNCATE_OLDEST = "truncate_oldest"


@dataclass
class ValidationResult:
    """Result of agent configuration validation.

    Contains validation outcome with categorized issues: errors prevent
    operation, warnings indicate potential concerns that do not block execution.

    Attributes:
        valid: True if no errors were found.
        errors: List of validation error descriptions.
        warnings: List of validation warning descriptions.
    """

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class Runnable(Protocol):
    """Protocol for objects that can be run with a BaseModel input.

    Both QuantedAgent and future workflow types (Pipeline, Router, Loop)
    implement this protocol, enabling uniform composition in Phase 3.
    """

    async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
        """Run the agent or workflow with the given input.

        Args:
            input_data: A Pydantic BaseModel instance as input.
            **kwargs: Additional keyword arguments passed to the underlying runner.

        Returns:
            A QuantedResult wrapping the validated output.
        """
        ...


InterceptorFn = Union[
    Callable[[str, dict[str, Any]], dict[str, Any] | None],
    Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | None]],
]
"""Type alias for MCP tool argument interceptor functions.

An interceptor receives the tool name and arguments dict, and returns
modified arguments or None to abort the tool call. May be sync or async.
"""


@runtime_checkable
class ConcurrencyBackend(Protocol):
    """Protocol for pluggable concurrency control backends.

    Implementations must provide an ``acquire`` method that returns an async
    context manager. The context manager acquires a slot on entry and releases
    it on exit.

    Example:
        async with backend.acquire(tool_name):
            result = await call_tool(tool_name, args)
    """

    def acquire(self, tool_name: str) -> Any:
        """Return an async context manager that acquires and releases a concurrency slot.

        Args:
            tool_name: The name of the tool requesting a slot.

        Returns:
            An async context manager for slot acquisition.
        """
        ...


class SemaphoreBackend:
    """Default in-process concurrency backend using asyncio.BoundedSemaphore.

    Limits the number of concurrent MCP tool calls per server. Uses
    BoundedSemaphore to prevent accidental over-release bugs. Supports
    optional timeout with fail-open semantics: when acquire times out,
    the call proceeds without throttling.

    Attributes:
        _semaphore: The asyncio BoundedSemaphore instance.
        _timeout: Optional timeout in seconds for semaphore acquisition.
    """

    def __init__(self, max_concurrent: int, timeout: float | None = None) -> None:
        """Initialize the semaphore backend.

        Args:
            max_concurrent: Maximum number of concurrent tool calls allowed.
            timeout: Optional timeout in seconds for semaphore acquisition.
                None means wait forever. On timeout, call proceeds (fail-open).
        """
        self._semaphore: asyncio.BoundedSemaphore = asyncio.BoundedSemaphore(max_concurrent)
        self._timeout: float | None = timeout

    @asynccontextmanager
    async def acquire(self, tool_name: str) -> AsyncIterator[None]:
        """Acquire a concurrency slot, yielding control inside the context.

        If a timeout is configured and acquisition times out, the call
        proceeds without holding the semaphore (fail-open).

        Args:
            tool_name: The name of the tool requesting a slot.

        Yields:
            None when the slot is acquired or timeout occurred (fail-open).
        """
        acquired = True
        try:
            if self._timeout is not None:
                await asyncio.wait_for(self._semaphore.acquire(), timeout=self._timeout)
            else:
                await self._semaphore.acquire()
        except asyncio.TimeoutError:
            acquired = False
        try:
            yield
        finally:
            if acquired:
                self._semaphore.release()
