"""Custom exception classes for the quanted_agents package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

    from quanted_agents.artifact_store import ArtifactStore
    from quanted_agents.result import QuantedResult


class InvalidInputType(TypeError):
    """Raised when input_type is not a Pydantic BaseModel subclass.

    This exception inherits from TypeError because it represents a type
    contract violation at construction time, not a runtime error.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "input_type must be a Pydantic BaseModel subclass"
        super().__init__(message)


class InvalidOutputType(TypeError):
    """Raised when output_type is not a Pydantic BaseModel subclass.

    This exception inherits from TypeError because it represents a type
    contract violation at construction time, not a runtime error.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "output_type must be a Pydantic BaseModel subclass"
        super().__init__(message)


class RecoveryExhaustedError(RuntimeError):
    """Raised when the recovery budget has been exhausted.

    This exception inherits from RuntimeError because it represents a
    runtime failure when recovery attempts are depleted, not a type
    contract violation.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Recovery budget exhausted"
        super().__init__(message)


class PipelineTypeError(TypeError):
    """Raised when Pipeline step output type does not match next step input type.

    This exception inherits from TypeError because it represents a type
    contract violation at construction time between adjacent pipeline steps.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Pipeline type mismatch between steps"
        super().__init__(message)


class RoutingError(ValueError):
    """Raised when Router dispatcher selects an invalid specialist target.

    This exception inherits from ValueError because the dispatcher returned
    a target name that does not exist in the specialists dictionary.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Invalid routing target"
        super().__init__(message)


class MaxIterationsExceeded(RuntimeError):
    """Raised when Loop hits max_iterations without termination.

    This exception inherits from RuntimeError because it represents a
    runtime failure when the loop body does not converge within the
    allowed iteration budget.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Maximum iterations exceeded"
        super().__init__(message)


class AssemblyError(Exception):
    """Raised when an assembly function fails.

    Preserves the ArtifactStore and the last QuantedResult so the caller
    can inspect intermediate state and debug the assembly logic.

    Attributes:
        store: The ArtifactStore containing all accumulated artifacts.
        last_result: The QuantedResult from the last orchestration step.
        original_error: The original exception raised by the assembly function.
    """

    def __init__(
        self,
        message: str,
        *,
        store: ArtifactStore,
        last_result: QuantedResult,
        original_error: Exception,
    ) -> None:
        super().__init__(message)
        self.store: ArtifactStore = store
        self.last_result: QuantedResult = last_result
        self.original_error: Exception = original_error


class AgentTimeoutError(TimeoutError):
    """Raised when an agent's hard timeout fires.

    Carries the artifact store and usage data so callers can inspect
    intermediate state even after a hard timeout cancellation.

    Attributes:
        store: The ArtifactStore containing artifacts committed before timeout.
        usage: Token usage statistics accumulated before timeout.
        termination_reason: Always "hard_timeout" for this exception.
    """

    def __init__(
        self,
        message: str,
        *,
        store: ArtifactStore | None,
        usage: RunUsage,
        termination_reason: str,
    ) -> None:
        super().__init__(message)
        self.store: ArtifactStore | None = store
        self.usage: RunUsage = usage
        self.termination_reason: str = termination_reason


class ContextOverflowError(Exception):
    """Raised when context window token count exceeds max_context_tokens.

    Carries token counts, artifact store, and usage so callers can decide
    how to handle the overflow (e.g., switch to truncation strategy).

    Attributes:
        current_tokens: The estimated token count that triggered the overflow.
        max_tokens: The configured maximum token limit.
        store: The ArtifactStore containing accumulated artifacts.
        usage: Token usage statistics accumulated before the overflow.
    """

    def __init__(
        self,
        *,
        current_tokens: int,
        max_tokens: int,
        store: ArtifactStore | None,
        usage: RunUsage,
    ) -> None:
        message = (
            f"Context overflow: {current_tokens} tokens exceeds "
            f"maximum of {max_tokens} tokens"
        )
        super().__init__(message)
        self.current_tokens: int = current_tokens
        self.max_tokens: int = max_tokens
        self.store: ArtifactStore | None = store
        self.usage: RunUsage = usage


class ConfigurationError(ValueError):
    """Raised when agent configuration validation fails.

    Carries the list of specific validation errors so callers can
    programmatically inspect what went wrong.

    Attributes:
        errors: List of human-readable error descriptions.
    """

    def __init__(self, errors: list[str]) -> None:
        message = "; ".join(errors)
        super().__init__(message)
        self.errors: list[str] = errors


class MCPConnectionError(ConnectionError):
    """Raised when a connection to an MCP server fails.

    This exception inherits from ConnectionError (built-in) because it
    represents a network-level failure to establish or maintain a connection
    to an MCP server endpoint.
    """

    def __init__(self, message: str = "") -> None:
        if not message:
            message = "Failed to connect to MCP server"
        super().__init__(message)
