"""Tests for Parallel branch retry feature (ORCH-02).

Validates that Parallel supports RetryPolicy for retrying failed branches,
only retries matching exception types, preserves successful branches,
and handles exhausted retries correctly.
"""

import json
import unittest
from typing import Any
from unittest.mock import AsyncMock, patch

from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from quanted_agents import QuantedAgent
from quanted_agents.result import QuantedResult
from quanted_agents.workflows import Parallel
from quanted_agents.workflows.parallel import ParallelResult, RetryPolicy


class RetryInput(BaseModel):
    """Input model for retry testing."""

    text: str


class RetryOutput(BaseModel):
    """Output model for retry testing."""

    value: str


class TransientError(Exception):
    """A transient error eligible for retry."""

    pass


class PermanentError(Exception):
    """A permanent error not eligible for retry."""

    pass


def _success_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """FunctionModel handler returning a successful RetryOutput."""
    output = RetryOutput(value="success")
    return ModelResponse(parts=[TextPart(content=json.dumps(output.model_dump()))])


class TestParallelRetry(unittest.IsolatedAsyncioTestCase):
    """Tests for Parallel retry_policy parameter."""

    async def test_no_retry_policy_unchanged_behavior(self) -> None:
        """Parallel without retry_policy behaves exactly as before."""
        call_count = 0

        def counting_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            return _success_model(messages, info)

        branch_a = QuantedAgent(
            "test", input_type=RetryInput, output_type=RetryOutput, system_prompt="A"
        )
        branch_b = QuantedAgent(
            "test", input_type=RetryInput, output_type=RetryOutput, system_prompt="B"
        )
        parallel = Parallel(branches=[branch_a, branch_b])

        with branch_a.inner.override(model=FunctionModel(counting_model)):
            with branch_b.inner.override(model=FunctionModel(counting_model)):
                result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(call_count, 2)

    async def test_retry_recovers_failed_branch(self) -> None:
        """One branch fails on first attempt but succeeds on retry."""
        call_count = 0

        class FailOnceRunnable:
            """A Runnable that fails on the first call, succeeds after."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TransientError("temporary failure")
                return QuantedResult.from_data(RetryOutput(value="recovered"))

        branch_a = FailOnceRunnable()

        class SuccessRunnable:
            """A Runnable that always succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                return QuantedResult.from_data(RetryOutput(value="ok"))

        branch_b = SuccessRunnable()

        policy = RetryPolicy(max_retries=2, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(branches=[branch_a, branch_b], retry_policy=policy)
        result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(call_count, 2)

    async def test_retry_only_retries_matching_exceptions(self) -> None:
        """Branch failing with non-matching exception type is not retried."""
        call_count = 0

        class PermanentFailRunnable:
            """A Runnable that fails with a permanent error."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal call_count
                call_count += 1
                raise PermanentError("permanent failure")

        class SuccessRunnable:
            """A Runnable that always succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                return QuantedResult.from_data(RetryOutput(value="ok"))

        policy = RetryPolicy(max_retries=3, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(
            branches=[PermanentFailRunnable(), SuccessRunnable()], retry_policy=policy
        )
        result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(result.results), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIsInstance(result.errors[0], PermanentError)
        self.assertEqual(call_count, 1)

    async def test_successful_branches_not_rerun(self) -> None:
        """Track call counts per branch, verify successful branch called exactly once."""
        success_count = 0
        fail_count = 0

        class SuccessCountRunnable:
            """A Runnable that counts successful calls."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal success_count
                success_count += 1
                return QuantedResult.from_data(RetryOutput(value="ok"))

        class FailOnceRunnable:
            """A Runnable that fails once then succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal fail_count
                fail_count += 1
                if fail_count == 1:
                    raise TransientError("temp")
                return QuantedResult.from_data(RetryOutput(value="recovered"))

        policy = RetryPolicy(max_retries=2, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(
            branches=[FailOnceRunnable(), SuccessCountRunnable()], retry_policy=policy
        )
        result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(success_count, 1)
        self.assertEqual(fail_count, 2)
        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(result.errors), 0)

    async def test_max_retries_exhausted(self) -> None:
        """Branch fails every time, error preserved after max_retries."""
        call_count = 0

        class AlwaysFailRunnable:
            """A Runnable that always fails."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal call_count
                call_count += 1
                raise TransientError(f"failure #{call_count}")

        class SuccessRunnable:
            """A Runnable that always succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                return QuantedResult.from_data(RetryOutput(value="ok"))

        policy = RetryPolicy(max_retries=3, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(
            branches=[AlwaysFailRunnable(), SuccessRunnable()], retry_policy=policy
        )
        result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(result.results), 1)
        self.assertEqual(len(result.errors), 1)
        self.assertIsInstance(result.errors[0], TransientError)
        # 1 initial + 3 retries = 4 total calls
        self.assertEqual(call_count, 4)

    async def test_retry_delay_applied(self) -> None:
        """Verify delay_seconds is used between retry attempts."""
        call_count = 0

        class FailTwiceRunnable:
            """A Runnable that fails twice then succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise TransientError("temp")
                return QuantedResult.from_data(RetryOutput(value="ok"))

        class SuccessRunnable:
            """A Runnable that always succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                return QuantedResult.from_data(RetryOutput(value="ok"))

        policy = RetryPolicy(max_retries=3, retry_on=[TransientError], delay_seconds=0.5)
        parallel = Parallel(
            branches=[FailTwiceRunnable(), SuccessRunnable()], retry_policy=policy
        )

        with patch("quanted_agents.workflows.parallel.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(result.results), 2)
        # First retry attempt (attempt=0) does not sleep, second (attempt=1) does
        self.assertEqual(mock_sleep.call_count, 1)
        mock_sleep.assert_called_with(0.5)

    async def test_retry_prepends_error_context(self) -> None:
        """Verify the retry call includes message_history with error context."""
        captured_kwargs: list[dict[str, Any]] = []
        call_count = 0

        class FailOnceCapturingRunnable:
            """A Runnable that fails once and captures kwargs on retry."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise TransientError("specific error message")
                captured_kwargs.append(kwargs)
                return QuantedResult.from_data(RetryOutput(value="ok"))

        class SuccessRunnable:
            """A Runnable that always succeeds."""

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                return QuantedResult.from_data(RetryOutput(value="ok"))

        policy = RetryPolicy(max_retries=1, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(
            branches=[FailOnceCapturingRunnable(), SuccessRunnable()], retry_policy=policy
        )
        await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(captured_kwargs), 1)
        history = captured_kwargs[0].get("message_history")
        self.assertIsNotNone(history)
        self.assertEqual(len(history), 1)
        content = history[0].parts[0].content
        self.assertIn("TransientError", content)
        self.assertIn("specific error message", content)

    async def test_multiple_failed_branches_retried_independently(self) -> None:
        """Two branches fail, both get retried independently."""
        call_counts = [0, 0]

        class FailOnceBranch:
            """A Runnable that fails once, identified by index."""

            def __init__(self, idx: int) -> None:
                self.idx: int = idx

            async def run(self, input_data: BaseModel, **kwargs: Any) -> QuantedResult[Any]:
                call_counts[self.idx] += 1
                if call_counts[self.idx] == 1:
                    raise TransientError(f"branch {self.idx} failed")
                return QuantedResult.from_data(RetryOutput(value=f"branch_{self.idx}"))

        policy = RetryPolicy(max_retries=2, retry_on=[TransientError], delay_seconds=0)
        parallel = Parallel(branches=[FailOnceBranch(0), FailOnceBranch(1)], retry_policy=policy)
        result = await parallel.run(RetryInput(text="test"))

        self.assertEqual(len(result.results), 2)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(call_counts[0], 2)
        self.assertEqual(call_counts[1], 2)


if __name__ == "__main__":
    unittest.main()
