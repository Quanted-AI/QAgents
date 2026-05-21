"""Tests for the error recovery pipeline module.

Validates json-repair integration, RecoveryBudget enforcement, and
RecoveryPipeline orchestration. All tests are synchronous since the
recovery module uses no async operations.
"""

import unittest

from quanted_agents.exceptions import RecoveryExhaustedError
from quanted_agents.recovery import RecoveryBudget, RecoveryPipeline, attempt_json_repair
from tests.conftest import (
    IRREPARABLE_TEXT,
    MALFORMED_EXTRA_TEXT,
    MALFORMED_MISSING_QUOTES,
    MALFORMED_SINGLE_QUOTES,
    MALFORMED_TRAILING_COMMA,
    VALID_JSON,
    SampleOutput,
)


class TestAttemptJsonRepair(unittest.TestCase):
    """Tests for the attempt_json_repair function.

    Verifies that common LLM JSON errors are repaired and validated
    against a Pydantic BaseModel schema, and that irreparable input
    returns None.
    """

    def test_repairs_missing_quotes(self) -> None:
        """Malformed JSON with missing quotes is repaired to a valid model."""
        result = attempt_json_repair(MALFORMED_MISSING_QUOTES, SampleOutput)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "test")

    def test_repairs_trailing_comma(self) -> None:
        """Malformed JSON with a trailing comma is repaired to a valid model."""
        result = attempt_json_repair(MALFORMED_TRAILING_COMMA, SampleOutput)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "test")

    def test_repairs_single_quotes(self) -> None:
        """Malformed JSON with single quotes is repaired to a valid model."""
        result = attempt_json_repair(MALFORMED_SINGLE_QUOTES, SampleOutput)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "test")

    def test_repairs_extra_text_around_json(self) -> None:
        """Malformed JSON with surrounding text is repaired to a valid model."""
        result = attempt_json_repair(MALFORMED_EXTRA_TEXT, SampleOutput)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "test")

    def test_valid_json_passes_through(self) -> None:
        """Already-valid JSON is parsed and validated successfully."""
        result = attempt_json_repair(VALID_JSON, SampleOutput)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "test")

    def test_returns_none_for_irreparable_text(self) -> None:
        """Plain text with no JSON structure returns None."""
        result = attempt_json_repair(IRREPARABLE_TEXT, SampleOutput)
        self.assertIsNone(result)

    def test_returns_none_for_empty_string(self) -> None:
        """An empty string returns None."""
        result = attempt_json_repair("", SampleOutput)
        self.assertIsNone(result)

    def test_returns_none_for_schema_mismatch(self) -> None:
        """Valid JSON with wrong fields returns None (Pydantic validation fails)."""
        result = attempt_json_repair('{"wrong_field": 42}', SampleOutput)
        self.assertIsNone(result)

    def test_returns_correct_basemodel_type(self) -> None:
        """Repaired result is an instance of the target BaseModel subclass."""
        result = attempt_json_repair(VALID_JSON, SampleOutput)
        self.assertIsInstance(result, SampleOutput)

    def test_extracts_values_correctly(self) -> None:
        """Repaired result contains the correct field values."""
        result = attempt_json_repair(VALID_JSON, SampleOutput)
        self.assertIsNotNone(result)
        self.assertEqual(result.answer, "test")
        self.assertAlmostEqual(result.confidence, 0.9)


class TestRecoveryBudget(unittest.TestCase):
    """Tests for the RecoveryBudget class.

    Verifies budget initialization, consumption, exhaustion behavior,
    and reset functionality.
    """

    def test_initial_remaining_equals_max(self) -> None:
        """A fresh budget has remaining equal to max_attempts."""
        budget = RecoveryBudget(3)
        self.assertEqual(budget.remaining, 3)

    def test_consume_decrements_remaining(self) -> None:
        """Consuming one attempt decrements the remaining count."""
        budget = RecoveryBudget(3)
        budget.consume()
        self.assertEqual(budget.remaining, 2)

    def test_raises_after_max_attempts(self) -> None:
        """Consuming more than max_attempts raises RecoveryExhaustedError."""
        budget = RecoveryBudget(3)
        budget.consume()
        budget.consume()
        budget.consume()
        with self.assertRaises(RecoveryExhaustedError):
            budget.consume()

    def test_raises_recovery_exhausted_error_type(self) -> None:
        """The raised exception is specifically RecoveryExhaustedError."""
        budget = RecoveryBudget(1)
        budget.consume()
        try:
            budget.consume()
            self.fail("Expected RecoveryExhaustedError to be raised")
        except RecoveryExhaustedError as exc:
            self.assertIsInstance(exc, RecoveryExhaustedError)
            self.assertIn("exhausted", str(exc).lower())

    def test_custom_max_attempts(self) -> None:
        """A custom max_attempts allows that many consumptions before raising."""
        budget = RecoveryBudget(5)
        for _ in range(5):
            budget.consume()
        with self.assertRaises(RecoveryExhaustedError):
            budget.consume()

    def test_reset_restores_budget(self) -> None:
        """Resetting the budget restores remaining to max_attempts."""
        budget = RecoveryBudget(3)
        budget.consume()
        budget.consume()
        self.assertEqual(budget.remaining, 1)
        budget.reset()
        self.assertEqual(budget.remaining, 3)


class TestRecoveryPipeline(unittest.TestCase):
    """Tests for the RecoveryPipeline class.

    Verifies that the pipeline orchestrates json-repair, Pydantic validation,
    and budget tracking correctly.
    """

    def test_attempt_repair_returns_basemodel_on_success(self) -> None:
        """Successful repair returns a validated BaseModel instance."""
        pipeline = RecoveryPipeline(output_type=SampleOutput, max_attempts=3)
        result = pipeline.attempt_repair(MALFORMED_SINGLE_QUOTES)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SampleOutput)

    def test_attempt_repair_returns_none_on_failure(self) -> None:
        """Failed repair returns None for irreparable input."""
        pipeline = RecoveryPipeline(output_type=SampleOutput, max_attempts=3)
        result = pipeline.attempt_repair(IRREPARABLE_TEXT)
        self.assertIsNone(result)

    def test_attempt_repair_consumes_budget(self) -> None:
        """Each attempt_repair call consumes one budget unit."""
        pipeline = RecoveryPipeline(output_type=SampleOutput, max_attempts=3)
        self.assertEqual(pipeline.budget.remaining, 3)
        pipeline.attempt_repair(VALID_JSON)
        self.assertEqual(pipeline.budget.remaining, 2)

    def test_pipeline_respects_budget_limit(self) -> None:
        """Pipeline raises RecoveryExhaustedError after budget is exceeded."""
        pipeline = RecoveryPipeline(output_type=SampleOutput, max_attempts=2)
        pipeline.attempt_repair(IRREPARABLE_TEXT)
        pipeline.attempt_repair(IRREPARABLE_TEXT)
        with self.assertRaises(RecoveryExhaustedError):
            pipeline.attempt_repair(IRREPARABLE_TEXT)

    def test_pipeline_with_output_type(self) -> None:
        """Pipeline validates against the configured output_type."""
        pipeline = RecoveryPipeline(output_type=SampleOutput, max_attempts=3)
        result = pipeline.attempt_repair(VALID_JSON)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, SampleOutput)
        self.assertEqual(result.answer, "test")
        self.assertAlmostEqual(result.confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
