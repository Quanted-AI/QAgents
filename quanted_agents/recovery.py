"""Error recovery pipeline for malformed LLM JSON output.

Provides a linear recovery sequence for handling invalid JSON from LLM responses:
1. Attempt to repair malformed JSON using json-repair
2. Validate the repaired data against a Pydantic BaseModel schema
3. Track recovery attempts via a budget to prevent infinite loops

The pipeline is designed to be integrated into QuantedAgent.run() (Phase 2, Plan 02)
as an intermediate step between receiving raw LLM output and returning validated results.
"""

from __future__ import annotations

import logging

from json_repair import repair_json
from pydantic import BaseModel, ValidationError

from quanted_agents.exceptions import RecoveryExhaustedError

logger = logging.getLogger(__name__)


def attempt_json_repair[T: BaseModel](raw_text: str, output_type: type[T]) -> T | None:
    """Attempt to repair malformed JSON and validate against a Pydantic model.

    Uses json-repair to fix common JSON issues (missing quotes, trailing commas,
    unescaped characters, etc.) then validates the repaired data against the
    provided Pydantic BaseModel subclass.

    Args:
        raw_text: The raw (potentially malformed) JSON string from LLM output.
        output_type: The Pydantic BaseModel subclass to validate against.

    Returns:
        A validated instance of output_type if repair and validation succeed,
        or None if the input is irreparable or fails validation.
    """
    try:
        repaired = repair_json(raw_text, return_objects=True)
    except Exception:
        logger.debug(f"json-repair failed on input: {raw_text[:200]}")
        return None

    if not isinstance(repaired, dict):
        logger.debug(f"json-repair returned non-dict type: {type(repaired).__name__}")
        return None

    try:
        return output_type.model_validate(repaired)
    except ValidationError as exc:
        logger.debug(f"Pydantic validation failed after repair: {exc}")
        return None


class RecoveryBudget:
    """Tracks recovery attempts and prevents infinite retry loops.

    Maintains a counter of consumed attempts against a configurable maximum.
    When the budget is exceeded, raises RecoveryExhaustedError to signal
    that no more recovery attempts should be made.

    Example:
        budget = RecoveryBudget(max_attempts=3)
        budget.consume()  # attempts: 1
        budget.consume()  # attempts: 2
        budget.consume()  # attempts: 3
        budget.consume()  # raises RecoveryExhaustedError
    """

    def __init__(self, max_attempts: int = 3) -> None:
        """Initialize the recovery budget.

        Args:
            max_attempts: Maximum number of recovery attempts allowed.
        """
        self.max_attempts: int = max_attempts
        self.attempts: int = 0

    def consume(self) -> None:
        """Consume one recovery attempt from the budget.

        Raises:
            RecoveryExhaustedError: If the budget has been exceeded after
                incrementing the attempt counter.
        """
        self.attempts += 1
        if self.attempts > self.max_attempts:
            raise RecoveryExhaustedError(
                f"Recovery budget exhausted after {self.max_attempts} attempts"
            )

    @property
    def remaining(self) -> int:
        """Return the number of remaining recovery attempts.

        Returns:
            The number of attempts still available, minimum 0.
        """
        return max(0, self.max_attempts - self.attempts)

    def reset(self) -> None:
        """Reset the attempt counter to zero.

        Useful for testing or reusing a budget across multiple operations.
        """
        self.attempts = 0


class RecoveryPipeline:
    """Orchestrates the linear recovery sequence for malformed LLM output.

    Combines json-repair and Pydantic validation with a recovery budget
    to provide a controlled, bounded approach to fixing LLM output errors.

    The pipeline follows a linear sequence for each attempt:
    1. Try to repair the raw JSON text
    2. Validate the repaired data against the output schema
    3. Consume one budget attempt regardless of success or failure

    Example:
        pipeline = RecoveryPipeline(output_type=MyModel, max_attempts=3)
        result = pipeline.attempt_repair('{"x": 42}')
        if result is not None:
            print(result)  # validated MyModel instance
    """

    def __init__(self, output_type: type[BaseModel], max_attempts: int = 3) -> None:
        """Initialize the recovery pipeline.

        Args:
            output_type: The Pydantic BaseModel subclass to validate repaired
                JSON against.
            max_attempts: Maximum number of recovery attempts before raising
                RecoveryExhaustedError.
        """
        self._output_type: type[BaseModel] = output_type
        self._budget: RecoveryBudget = RecoveryBudget(max_attempts)

    def attempt_repair(self, raw_text: str) -> BaseModel | None:
        """Attempt to repair and validate raw JSON text.

        Tries to repair the JSON, validates against the output type schema,
        and consumes one budget attempt. Both successful and failed repairs
        consume budget.

        Args:
            raw_text: The raw (potentially malformed) JSON string.

        Returns:
            A validated BaseModel instance if repair and validation succeed,
            or None if the repair failed.

        Raises:
            RecoveryExhaustedError: If the recovery budget has been exceeded.
        """
        self._budget.consume()
        result = attempt_json_repair(raw_text, self._output_type)
        return result

    @property
    def budget(self) -> RecoveryBudget:
        """Expose the recovery budget for external inspection.

        Returns:
            The internal RecoveryBudget instance.
        """
        return self._budget
