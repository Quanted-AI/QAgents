"""Tool Middleware Example: Argument Interceptor and Concurrency Control.

Demonstrates MCPTool middleware features: argument_interceptor for modifying
tool arguments before execution, max_concurrent_calls for throttling, and
tool_retry_max for automatic retry with backoff.

The middleware pipeline chains in fixed order:
intercept -> throttle -> execute (with retry) -> trace.

NOTE: This example requires a running MCP server. It shows the configuration
pattern. Without an MCP server, the agent will not be able to call tools,
but the agent creation and middleware setup will succeed.

Requires an LLM API key. Set the appropriate environment variable for your
chosen provider (e.g., OPENAI_API_KEY).
"""

# Requires: OPENAI_API_KEY environment variable
# Requires: A running MCP server at the configured URL (for tool execution)

import asyncio
import os
import sys
from typing import Any

from pydantic import BaseModel

from quanted_agents import MCPTool, QuantedAgent


# Default model -- override with QUANTED_MODEL env var
MODEL = os.environ.get("QUANTED_MODEL", "openai:gpt-4o-mini")

# MCP server URL -- override with MCP_SERVER_URL env var
MCP_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:8001/mcp")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Query(BaseModel):
    """Input for the agent.

    Attributes:
        question: The question to answer using tools.
    """

    question: str


class Answer(BaseModel):
    """Output from the agent.

    Attributes:
        response: The answer text.
        tools_used: List of tools that were called.
    """

    response: str
    tools_used: list[str]


# ---------------------------------------------------------------------------
# Pattern 1: Argument Interceptor
# ---------------------------------------------------------------------------

def add_tenant_id(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Interceptor that adds a tenant_id to all tool arguments.

    This is a common pattern for multi-tenant applications where every
    tool call needs to include the current tenant context.

    Args:
        tool_name: The name of the tool being called.
        args: The original tool arguments.

    Returns:
        Modified arguments with tenant_id added.
    """
    return {**args, "tenant_id": "tenant_abc123"}


async def interceptor_example() -> None:
    """Demonstrate argument interceptor on MCPTool."""
    print("=" * 60)
    print("Pattern 1: Argument Interceptor")
    print("=" * 60)

    # MCPTool with interceptor -- adds tenant_id to every tool call
    tools = MCPTool(
        MCP_URL,
        argument_interceptor=add_tenant_id,
    )

    agent = QuantedAgent(
        MODEL,
        input_type=Query,
        output_type=Answer,
        system_prompt="Answer questions using available tools.",
        toolsets=[tools],
    )

    print(f"MCPTool configured with interceptor at {MCP_URL}")
    print("Interceptor adds tenant_id='tenant_abc123' to every tool call")
    print()


# ---------------------------------------------------------------------------
# Pattern 2: Concurrency Control + Retry
# ---------------------------------------------------------------------------

async def concurrency_retry_example() -> None:
    """Demonstrate concurrency throttling and retry on MCPTool."""
    print("=" * 60)
    print("Pattern 2: Concurrency + Retry")
    print("=" * 60)

    # MCPTool with concurrency limit and retry
    tools = MCPTool(
        MCP_URL,
        max_concurrent_calls=3,        # At most 3 concurrent tool calls
        tool_retry_max=2,              # Retry failed calls up to 2 times
        tool_retry_delay=1.0,          # 1 second base delay between retries
        tool_retry_backoff_factor=2.0,  # Exponential backoff: 1s, 2s
    )

    agent = QuantedAgent(
        MODEL,
        input_type=Query,
        output_type=Answer,
        system_prompt="Answer questions using available tools.",
        toolsets=[tools],
    )

    print(f"MCPTool configured with concurrency=3, retry=2 at {MCP_URL}")
    print("Retry delays: 1.0s, 2.0s (exponential backoff)")
    print()


# ---------------------------------------------------------------------------
# Pattern 3: Full Middleware Stack
# ---------------------------------------------------------------------------

def security_interceptor(tool_name: str, args: dict[str, Any]) -> dict[str, Any] | None:
    """Interceptor that enforces security rules.

    Blocks tool calls with potentially unsafe arguments and adds
    audit metadata to all calls.

    Args:
        tool_name: The name of the tool being called.
        args: The original tool arguments.

    Returns:
        Modified arguments with audit metadata, or None to abort.
    """
    # Block dangerous patterns
    blocked_patterns = ["rm -rf", "DROP TABLE", "DELETE FROM"]
    args_str = str(args)
    for pattern in blocked_patterns:
        if pattern in args_str:
            return None  # Abort -- interceptor returns None

    # Add audit metadata
    return {**args, "_audit_tool": tool_name, "_audit_safe": True}


async def full_middleware_example() -> None:
    """Demonstrate the full middleware stack: intercept + throttle + retry + trace."""
    print("=" * 60)
    print("Pattern 3: Full Middleware Stack")
    print("=" * 60)

    # Full middleware pipeline
    tools = MCPTool(
        MCP_URL,
        argument_interceptor=security_interceptor,  # Stage 1: intercept
        max_concurrent_calls=5,                      # Stage 2: throttle
        tool_retry_max=3,                            # Stage 3: retry on execute
        tool_retry_delay=0.5,
        tool_retry_backoff_factor=2.0,
        tool_trace_level="standard",                 # Stage 4: trace
    )

    agent = QuantedAgent(
        MODEL,
        input_type=Query,
        output_type=Answer,
        system_prompt="Answer questions using available tools.",
        toolsets=[tools],
    )

    print(f"MCPTool configured with full middleware stack at {MCP_URL}")
    print("Pipeline: intercept -> throttle(5) -> execute(retry=3) -> trace(standard)")
    print()

    # Show middleware configuration summary
    print("Middleware configuration:")
    print(f"  Interceptor: security_interceptor (blocks dangerous patterns)")
    print(f"  Concurrency: max 5 concurrent calls")
    print(f"  Retry: up to 3 attempts, delays 0.5s/1.0s/2.0s")
    print(f"  Tracing: standard level (args + result preview)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    """Run all three middleware patterns."""
    await interceptor_example()
    await concurrency_retry_example()
    await full_middleware_example()

    print("All patterns configured.")
    print("Note: To execute tool calls, start an MCP server at the configured URL.")


if __name__ == "__main__":
    api_key_vars = [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ]
    has_key = any(os.environ.get(var) for var in api_key_vars)
    if not has_key:
        print(
            "No API key found. Set one of the following environment variables:\n"
            f"  {', '.join(api_key_vars)}\n"
            "Then run: python examples/11_tool_middleware.py"
        )
        sys.exit(1)

    asyncio.run(main())
