"""Tests for MCP integration: MCPTool factory, QuantedAgent toolsets, MCPConnectionError, and retry.

Validates MCPTool creates correct pydantic-ai toolset instances for both Streamable HTTP
and SSE transports, parameter forwarding, QuantedAgent toolsets acceptance, async context
manager protocol, MCPConnectionError exception behavior, and transparent tool call retry
with async backoff. All tests use pydantic-ai's TestModel to avoid real LLM/MCP server
calls. MCP server connections are mocked via unittest.mock to prevent network access
during testing.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, call, patch

from pydantic_ai.mcp import MCPServerSSE, MCPServerStreamableHTTP
from pydantic_ai.models.test import TestModel

from quanted_agents import MCPConnectionError, MCPTool, QuantedAgent, QuantedResult
from quanted_agents.mcp import _retrying_direct_call_tool
from tests.conftest import SampleInput, SampleOutput


class TestMCPToolCreation(unittest.TestCase):
    """Tests for MCPTool factory function instantiation and parameter forwarding."""

    def test_creates_streamable_http_by_default(self) -> None:
        """MCPTool with no transport argument returns MCPServerStreamableHTTP."""
        result = MCPTool("http://localhost:8001/mcp")
        self.assertIsInstance(result, MCPServerStreamableHTTP)

    def test_creates_sse_with_transport_sse(self) -> None:
        """MCPTool with transport='sse' returns MCPServerSSE."""
        result = MCPTool("http://localhost:8001/mcp", transport="sse")
        self.assertIsInstance(result, MCPServerSSE)

    def test_invalid_transport_raises_valueerror(self) -> None:
        """MCPTool with an unrecognized transport value raises ValueError."""
        with self.assertRaises(ValueError) as ctx:
            MCPTool("http://x", transport="websocket")
        self.assertIn("websocket", str(ctx.exception))

    def test_tool_prefix_forwarded(self) -> None:
        """MCPTool forwards tool_prefix to the underlying MCPServer instance."""
        result = MCPTool("http://localhost:8001/mcp", tool_prefix="weather")
        self.assertEqual(result.tool_prefix, "weather")

    def test_timeout_forwarded(self) -> None:
        """MCPTool forwards timeout to the underlying MCPServer instance."""
        result = MCPTool("http://localhost:8001/mcp", timeout=30)
        self.assertEqual(result.timeout, 30)

    def test_url_stored(self) -> None:
        """MCPTool stores the URL on the returned MCPServer instance."""
        result = MCPTool("http://localhost:8001/mcp")
        self.assertEqual(result.url, "http://localhost:8001/mcp")


class TestQuantedAgentMCP(unittest.IsolatedAsyncioTestCase):
    """Tests for QuantedAgent integration with MCP toolsets parameter."""

    def test_agent_accepts_toolsets_parameter(self) -> None:
        """QuantedAgent constructs without error when given a toolsets list."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            toolsets=[MCPTool("http://localhost:8001/mcp")],
        )
        self.assertIsNotNone(agent)

    async def test_agent_without_toolsets_still_works(self) -> None:
        """QuantedAgent without toolsets runs normally (no regressions)."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="test"))
            self.assertIsInstance(result, QuantedResult)

    def test_agent_has_context_manager_protocol(self) -> None:
        """QuantedAgent implements __aenter__ and __aexit__ for async context management."""
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
        )
        self.assertTrue(hasattr(agent, "__aenter__"))
        self.assertTrue(hasattr(agent, "__aexit__"))

    @patch.object(MCPServerStreamableHTTP, "list_tools", new_callable=AsyncMock, return_value=[])
    @patch.object(MCPServerStreamableHTTP, "__aexit__", new_callable=AsyncMock)
    @patch.object(MCPServerStreamableHTTP, "__aenter__", new_callable=AsyncMock)
    async def test_agent_run_with_toolsets_and_test_model(
        self, mock_aenter: AsyncMock, mock_aexit: AsyncMock, mock_list_tools: AsyncMock
    ) -> None:
        """QuantedAgent with toolsets runs successfully using TestModel.

        MCP server connections and tool listing are mocked to avoid network
        access. This verifies toolsets do not interfere with normal agent
        execution when no MCP server is available.
        """
        toolset = MCPTool("http://localhost:8001/mcp")
        mock_aenter.return_value = toolset
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test agent",
            toolsets=[toolset],
        )
        with agent.inner.override(model=TestModel()):
            result = await agent.run(SampleInput(question="mcp test"))
            self.assertIsInstance(result, QuantedResult)
            self.assertIsInstance(result.data, SampleOutput)


class TestMCPConnectionError(unittest.TestCase):
    """Tests for MCPConnectionError exception class."""

    def test_mcp_connection_error_inherits_connection_error(self) -> None:
        """MCPConnectionError is a subclass of built-in ConnectionError."""
        self.assertTrue(issubclass(MCPConnectionError, ConnectionError))

    def test_mcp_connection_error_default_message(self) -> None:
        """MCPConnectionError has a meaningful default message when none is provided."""
        error = MCPConnectionError()
        self.assertEqual(str(error), "Failed to connect to MCP server")

    def test_mcp_connection_error_custom_message(self) -> None:
        """MCPConnectionError preserves a custom message when provided."""
        error = MCPConnectionError("Custom: server unreachable at port 8001")
        self.assertEqual(str(error), "Custom: server unreachable at port 8001")

    def test_importable_from_top_level(self) -> None:
        """MCPConnectionError is importable from quanted_agents top-level package."""
        from quanted_agents import MCPConnectionError as TopLevelError
        self.assertIs(TopLevelError, MCPConnectionError)


class TestMCPConnectionErrorRaised(unittest.IsolatedAsyncioTestCase):
    """Tests that QuantedAgent.__aenter__ raises MCPConnectionError on connection failure."""

    @patch.object(MCPServerStreamableHTTP, "__aenter__", new_callable=AsyncMock)
    async def test_agent_raises_mcp_connection_error_on_connect_failure(
        self, mock_aenter: AsyncMock
    ) -> None:
        """QuantedAgent wraps ConnectionError during __aenter__ as MCPConnectionError."""
        mock_aenter.side_effect = ConnectionError("Connection refused")
        toolset = MCPTool("http://localhost:8001/mcp")
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
            toolsets=[toolset],
        )
        with self.assertRaises(MCPConnectionError) as ctx:
            async with agent:
                pass
        self.assertIn("Connection refused", str(ctx.exception))

    @patch.object(MCPServerStreamableHTTP, "__aenter__", new_callable=AsyncMock)
    async def test_agent_raises_mcp_connection_error_on_os_error(
        self, mock_aenter: AsyncMock
    ) -> None:
        """QuantedAgent wraps OSError during __aenter__ as MCPConnectionError."""
        mock_aenter.side_effect = OSError("Network unreachable")
        toolset = MCPTool("http://localhost:8001/mcp")
        agent = QuantedAgent(
            "test",
            input_type=SampleInput,
            output_type=SampleOutput,
            system_prompt="Test",
            toolsets=[toolset],
        )
        with self.assertRaises(MCPConnectionError) as ctx:
            async with agent:
                pass
        self.assertIn("Network unreachable", str(ctx.exception))


class TestMCPToolExports(unittest.TestCase):
    """Tests for MCPTool import accessibility from package modules."""

    def test_mcp_tool_importable_from_top_level(self) -> None:
        """MCPTool is importable from quanted_agents top-level package."""
        from quanted_agents import MCPTool as TopLevelMCPTool
        self.assertIs(TopLevelMCPTool, MCPTool)

    def test_mcp_tool_importable_from_mcp_module(self) -> None:
        """MCPTool is importable from quanted_agents.mcp submodule."""
        from quanted_agents.mcp import MCPTool as ModuleMCPTool
        self.assertIs(ModuleMCPTool, MCPTool)


class TestMCPToolRetry(unittest.IsolatedAsyncioTestCase):
    """Tests for MCP tool call retry with async backoff behavior."""

    def test_default_no_retry_does_not_wrap(self) -> None:
        """MCPTool with no retry args does not wrap direct_call_tool."""
        server = MCPTool("http://localhost:8001/mcp")
        self.assertFalse(hasattr(server.direct_call_tool, "__wrapped__"))

    def test_retry_wraps_direct_call_tool(self) -> None:
        """MCPTool with tool_retry_max>0 wraps direct_call_tool with retry logic."""
        server = MCPTool("http://localhost:8001/mcp", tool_retry_max=2)
        self.assertTrue(hasattr(server.direct_call_tool, "__wrapped__"))

    async def test_retry_succeeds_on_first_attempt(self) -> None:
        """Wrapped tool call returns immediately on first successful attempt."""
        mock_fn = AsyncMock(return_value="success")
        result = await _retrying_direct_call_tool(
            mock_fn, "tool_name", {"arg": "val"}, None,
            max_retries=3, base_delay=0.0, backoff_factor=1.0,
        )
        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 1)

    async def test_retry_succeeds_after_transient_failures(self) -> None:
        """Wrapped tool call retries and succeeds after transient failures."""
        mock_fn = AsyncMock(
            side_effect=[Exception("timeout"), Exception("timeout"), "success"]
        )
        result = await _retrying_direct_call_tool(
            mock_fn, "tool_name", {"arg": "val"}, None,
            max_retries=3, base_delay=0.0, backoff_factor=1.0,
        )
        self.assertEqual(result, "success")
        self.assertEqual(mock_fn.call_count, 3)

    async def test_retry_exhausted_raises_last_error(self) -> None:
        """After all retries exhausted, the last exception propagates."""
        mock_fn = AsyncMock(
            side_effect=[Exception("err1"), Exception("err2"), Exception("err3")]
        )
        with self.assertRaises(Exception) as ctx:
            await _retrying_direct_call_tool(
                mock_fn, "tool_name", {"arg": "val"}, None,
                max_retries=2, base_delay=0.0, backoff_factor=1.0,
            )
        self.assertEqual(str(ctx.exception), "err3")
        self.assertEqual(mock_fn.call_count, 3)

    @patch("quanted_agents.mcp.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_backoff_delay_calculation(self, mock_sleep: AsyncMock) -> None:
        """Exponential backoff calculates correct delays: 1.0, 2.0, 4.0."""
        mock_fn = AsyncMock(side_effect=Exception("fail"))
        with self.assertRaises(Exception):
            await _retrying_direct_call_tool(
                mock_fn, "tool_name", {"arg": "val"}, None,
                max_retries=3, base_delay=1.0, backoff_factor=2.0,
            )
        self.assertEqual(mock_sleep.call_count, 3)
        mock_sleep.assert_has_calls([
            call(1.0),   # 1.0 * 2^0
            call(2.0),   # 1.0 * 2^1
            call(4.0),   # 1.0 * 2^2
        ])

    @patch("quanted_agents.mcp.asyncio.sleep", new_callable=AsyncMock)
    async def test_retry_constant_delay(self, mock_sleep: AsyncMock) -> None:
        """Constant delay (backoff_factor=1.0) uses same delay for all retries."""
        mock_fn = AsyncMock(side_effect=Exception("fail"))
        with self.assertRaises(Exception):
            await _retrying_direct_call_tool(
                mock_fn, "tool_name", {"arg": "val"}, None,
                max_retries=2, base_delay=0.5, backoff_factor=1.0,
            )
        self.assertEqual(mock_sleep.call_count, 2)
        mock_sleep.assert_has_calls([call(0.5), call(0.5)])

    def test_retry_works_with_sse_transport(self) -> None:
        """MCPTool with transport='sse' and tool_retry_max>0 wraps direct_call_tool."""
        server = MCPTool(
            "http://localhost:8001/sse", transport="sse", tool_retry_max=2
        )
        self.assertIsInstance(server, MCPServerSSE)
        self.assertTrue(hasattr(server.direct_call_tool, "__wrapped__"))


if __name__ == "__main__":
    unittest.main()
