"""Router Example: Support Ticket Router.

Demonstrates a Router workflow with a dispatcher agent and specialist agents.
The dispatcher classifies incoming support tickets and routes them to the
appropriate specialist (billing, technical, or general support).

The Router flow is:
1. Dispatcher receives the ticket and returns a RoutingDecision
2. Router validates the target specialist exists
3. Selected specialist handles the ticket and returns the response
"""

# Requires: OPENAI_API_KEY environment variable

import asyncio

from pydantic import BaseModel

from quanted_agents import QuantedAgent, Router, RoutingDecision


class SupportTicket(BaseModel):
    """Input for the support ticket router.

    Attributes:
        subject: The ticket subject line.
        body: The full ticket description.
        priority: Ticket priority level ("low", "normal", "high", "urgent").
    """

    subject: str
    body: str
    priority: str = "normal"


class TicketResponse(BaseModel):
    """Output from a specialist agent handling a support ticket.

    Attributes:
        response: The support response text for the customer.
        category: The resolved category of the ticket.
        estimated_resolution: Estimated time to full resolution.
    """

    response: str
    category: str
    estimated_resolution: str


dispatcher = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=RoutingDecision,
    system_prompt=(
        "You are a support ticket classifier. Analyze the ticket and decide which "
        "specialist should handle it. Available specialists:\n"
        '- "billing": Payment issues, invoices, subscriptions, refunds\n'
        '- "technical": Software bugs, API errors, integration issues, performance\n'
        '- "general": Account questions, feature requests, general inquiries\n\n'
        "Return the specialist name as target and explain your reasoning."
    ),
)

billing_specialist = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=TicketResponse,
    system_prompt=(
        "You are a billing support specialist. Help customers with payment issues, "
        "invoice questions, subscription management, and refund requests. Be empathetic "
        "and provide clear resolution steps."
    ),
)

technical_specialist = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=TicketResponse,
    system_prompt=(
        "You are a technical support specialist. Help customers with software bugs, "
        "API errors, integration issues, and performance problems. Provide specific "
        "troubleshooting steps and code examples where helpful."
    ),
)

general_specialist = QuantedAgent(
    "openai:gpt-4o-mini",
    input_type=SupportTicket,
    output_type=TicketResponse,
    system_prompt=(
        "You are a general support specialist. Help customers with account questions, "
        "feature requests, and general inquiries. Be friendly and provide helpful "
        "guidance."
    ),
)

router = Router(
    dispatcher=dispatcher,
    specialists={
        "billing": billing_specialist,
        "technical": technical_specialist,
        "general": general_specialist,
    },
)


async def main() -> None:
    """Run the support ticket router on a sample billing ticket."""
    ticket = SupportTicket(
        subject="Double charged on my subscription",
        body=(
            "I was charged twice for my monthly Pro subscription on February 15th. "
            "My credit card shows two charges of $29.99 from your company. I need "
            "a refund for the duplicate charge as soon as possible."
        ),
        priority="high",
    )

    result = await router.run(ticket)

    # Access the specialist's response
    print("=== Ticket Response ===")
    print(f"Category: {result.data.category}")
    print(f"Response: {result.data.response}")
    print(f"Estimated Resolution: {result.data.estimated_resolution}")

    # Observability: aggregated usage (dispatcher + specialist)
    print("\n=== Aggregated Usage ===")
    print(f"Total input tokens: {result.total_usage.input_tokens}")
    print(f"Total output tokens: {result.total_usage.output_tokens}")
    print(f"Total requests: {result.total_usage.requests}")

    # Observability: per-step timing (dispatcher + specialist)
    print("\n=== Step Timings ===")
    for timing in result.step_timings:
        print(f"  {timing.step_name}: {timing.duration_seconds:.2f}s")

    # Observability: execution trace
    print("\n=== Trace ===")
    for entry in result.trace:
        print(f"  {entry.step_name}: {entry.timing.duration_seconds:.2f}s (model: {entry.model_name})")


if __name__ == "__main__":
    asyncio.run(main())
