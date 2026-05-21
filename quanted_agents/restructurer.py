"""Two-model restructurer agent for the recovery pipeline.

Implements the two-model pattern where a cheap model (e.g., GPT-4o-mini, Claude Haiku)
restructures the heavy model's raw output into a validated Pydantic BaseModel schema.

The RestructurerAgent wraps a pydantic-ai Agent configured with a system prompt
specifically tuned for JSON restructuring. Before making any LLM call, it first
attempts json-repair (ERRR-04: restructurer has its own recovery pipeline), only
falling back to the cheap model when mechanical repair fails.

Flow:
    1. Receive raw text from the heavy model
    2. Try json-repair + Pydantic validation (no LLM call needed if successful)
    3. If repair fails, delegate to the cheap model via pydantic-ai Agent
    4. Return the validated BaseModel instance

This module is used by QuantedAgent.run() when a ``restructurer_model`` is configured.
"""

from __future__ import annotations

import logging
from typing import Any, cast

from pydantic import BaseModel
from pydantic_ai import Agent

from quanted_agents.recovery import attempt_json_repair

logger = logging.getLogger(__name__)

_RESTRUCTURER_SYSTEM_PROMPT = (
    "You are a JSON restructuring assistant. Your ONLY job is to take the provided "
    "text and restructure it into the exact JSON schema requested. Extract relevant "
    "values from the input and map them to the correct fields. Output ONLY valid "
    "JSON, no explanations."
)


class RestructurerAgent:
    """Restructures raw LLM output into a validated Pydantic BaseModel using a cheap model.

    The restructurer first attempts json-repair on the raw text (defense-in-depth per
    ERRR-04). Only if mechanical repair fails does it invoke the cheap model. The cheap
    model runs via a pydantic-ai Agent with output_type enforcement, so its output is
    also validated against the target schema.

    Example:
        from quanted_agents.restructurer import RestructurerAgent

        restructurer = RestructurerAgent(
            model="openai:gpt-4o-mini",
            output_type=MyOutputModel,
            retries=1,
        )
        result = await restructurer.restructure("{'answer': 'hello', 'score': 42}")
        print(result)  # MyOutputModel(answer='hello', score=42)
    """

    def __init__(
        self,
        model: str,
        output_type: type[BaseModel],
        retries: int = 1,
    ) -> None:
        """Create a new RestructurerAgent.

        Args:
            model: pydantic-ai model identifier for the cheap model
                (e.g., "openai:gpt-4o-mini", "anthropic:claude-haiku").
            output_type: The Pydantic BaseModel subclass to restructure into.
            retries: Number of output validation retries for the internal
                pydantic-ai Agent (pydantic-ai's built-in re-prompting).
        """
        self._output_type: type[BaseModel] = output_type
        self._agent: Agent[None, Any] = Agent(
            model,
            output_type=output_type,
            system_prompt=_RESTRUCTURER_SYSTEM_PROMPT,
            retries=retries,
        )

    async def restructure(self, raw_text: str) -> BaseModel:
        """Restructure raw text into the target BaseModel schema.

        First attempts json-repair on the raw text. If mechanical repair succeeds,
        returns immediately without an LLM call. If repair fails, delegates to the
        internal pydantic-ai Agent which uses the cheap model to restructure the text.

        pydantic-ai exceptions (e.g., UnexpectedModelBehavior) are allowed to propagate
        so the caller can handle budget tracking.

        Args:
            raw_text: The raw (potentially malformed) text from the heavy model.

        Returns:
            A validated BaseModel instance matching the configured output_type.

        Raises:
            UnexpectedModelBehavior: If the cheap model also fails after its retries.
        """
        repaired = attempt_json_repair(raw_text, self._output_type)
        if repaired is not None:
            logger.debug("Restructurer: json-repair succeeded, skipping LLM call")
            return repaired

        logger.debug("Restructurer: json-repair failed, delegating to cheap model")
        result = await self._agent.run(raw_text)
        return cast(BaseModel, result.output)

    @property
    def inner(self) -> Agent[None, Any]:
        """Access the underlying pydantic-ai Agent.

        Escape hatch for testing (e.g., overriding the model with FunctionModel).

        Returns:
            The wrapped pydantic-ai Agent instance.
        """
        return self._agent
