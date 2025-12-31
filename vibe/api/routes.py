"""FastAPI routes for Vibe agent interaction.

This module provides optional FastAPI routes for web-based agent interaction.
FastAPI must be installed separately: `pip install fastapi uvicorn`

Usage:
    from fastapi import FastAPI
    from vibe.api.routes import create_agent_router
    from vibe.core.config import VibeConfig

    app = FastAPI()
    config = VibeConfig()
    router = create_agent_router(config)
    app.include_router(router, prefix="/api/agent")

    # Run with: uvicorn main:app
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any
from contextlib import asynccontextmanager

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Stub types for when FastAPI is not installed
    APIRouter = Any  # type: ignore
    HTTPException = Exception  # type: ignore
    Request = Any  # type: ignore
    StreamingResponse = Any  # type: ignore

from vibe.api.consumer import SSEEventConsumer
from vibe.api.events import EventSerializer
from vibe.core.dispatchers import EventDispatcher
from vibe.core.runners import AgentRunner, AutoApproveHandler
from vibe.ipc.event_bus import EventBusPublisher, default_event_bus_config

if TYPE_CHECKING:
    from vibe.core.agent import Agent
    from vibe.core.config import VibeConfig

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., description="User message to send to agent")
    stream: bool = Field(True, description="Whether to stream the response")


class ChatResponse(BaseModel):
    """Response body for non-streaming chat."""

    content: str = Field(..., description="Agent response content")
    tool_calls: int = Field(0, description="Number of tool calls made")
    tokens_used: int = Field(0, description="Tokens used in this turn")


class AgentStatus(BaseModel):
    """Agent status response."""

    running: bool = Field(..., description="Whether agent is currently running")
    session_tokens: int = Field(0, description="Total tokens used in session")
    tool_calls: int = Field(0, description="Total tool calls in session")


def create_agent_router(
    config: "VibeConfig",
    auto_approve: bool = False,
) -> "APIRouter":
    """Create a FastAPI router for agent interaction.

    Args:
        config: Vibe configuration
        auto_approve: Whether to auto-approve all tool calls

    Returns:
        FastAPI router with agent endpoints
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install fastapi uvicorn"
        )

    router = APIRouter()

    # Shared state
    agent: Agent | None = None
    dispatcher: EventDispatcher | None = None
    runner: AgentRunner | None = None
    event_bus_publisher: EventBusPublisher | None = None
    active_consumers: list[SSEEventConsumer] = []

    async def get_or_create_agent() -> tuple[Agent, EventDispatcher, AgentRunner]:
        """Get or lazily create the agent."""
        nonlocal agent, dispatcher, runner, event_bus_publisher

        if agent is None:
            from vibe.core.agent import Agent

            agent = Agent(config, auto_approve=auto_approve)
            bus_config = default_event_bus_config(config.effective_workdir)
            event_bus_publisher = EventBusPublisher(bus_config)
            logger.info(
                "ipc.event_bus_publisher_ready",
                extra={"address": event_bus_publisher.address, "context": "api"},
            )
            dispatcher = EventDispatcher(event_bus=event_bus_publisher)
            approval_handler = AutoApproveHandler(auto_approve)
            runner = AgentRunner(agent, dispatcher, approval_handler)

        return agent, dispatcher, runner  # type: ignore

    @router.get("/status")
    async def get_status() -> AgentStatus:
        """Get agent status."""
        if agent is None:
            return AgentStatus(running=False, session_tokens=0, tool_calls=0)

        return AgentStatus(
            running=runner.is_running if runner else False,
            session_tokens=agent.stats.session_total_llm_tokens,
            tool_calls=agent.stats.tool_calls_succeeded + agent.stats.tool_calls_failed,
        )

    @router.post("/chat")
    async def chat(request: ChatRequest) -> ChatResponse | StreamingResponse:
        """Send a message to the agent.

        If stream=true (default), returns SSE stream of events.
        If stream=false, returns complete response.
        """
        _, disp, run = await get_or_create_agent()

        if request.stream:
            # Create SSE consumer for this request
            consumer = SSEEventConsumer()
            disp.add_consumer(consumer)
            active_consumers.append(consumer)

            async def generate_events():
                try:
                    # Start agent in background
                    agent_task = asyncio.create_task(
                        _run_agent(run, request.message)
                    )

                    # Stream events
                    async for event_str in consumer.events():
                        yield event_str

                        # Check if agent is done
                        if agent_task.done():
                            # Drain remaining queue
                            while consumer.queue_size > 0:
                                try:
                                    event_str = await asyncio.wait_for(
                                        consumer._queue.get(), timeout=0.1
                                    )
                                    yield event_str
                                except asyncio.TimeoutError:
                                    break
                            break

                    # Wait for agent to complete
                    await agent_task

                    # Send completion event
                    yield f"event: done\ndata: {{}}\n\n"

                except Exception as e:
                    logger.error(f"SSE error: {e}")
                    yield f"event: error\ndata: {{\"error\": \"{str(e)}\"}}\n\n"

                finally:
                    consumer.close()
                    disp.remove_consumer(consumer)
                    if consumer in active_consumers:
                        active_consumers.remove(consumer)

            return StreamingResponse(
                generate_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        else:
            # Non-streaming: collect all events
            content_parts: list[str] = []
            tool_calls = 0

            async for event in run.run(request.message):
                from vibe.core.types import AssistantEvent, ToolCallEvent

                if isinstance(event, AssistantEvent):
                    content_parts.append(event.content)
                elif isinstance(event, ToolCallEvent):
                    tool_calls += 1

            return ChatResponse(
                content="".join(content_parts),
                tool_calls=tool_calls,
                tokens_used=agent.stats.last_turn_total_tokens if agent else 0,
            )

    @router.post("/interrupt")
    async def interrupt() -> dict[str, bool]:
        """Interrupt the running agent."""
        if runner and runner.is_running:
            await runner.interrupt()
            return {"interrupted": True}
        return {"interrupted": False}

    @router.post("/clear")
    async def clear_history() -> dict[str, bool]:
        """Clear agent conversation history."""
        if runner:
            await runner.clear_history()
            return {"cleared": True}
        return {"cleared": False}

    @router.post("/compact")
    async def compact_history() -> dict[str, str]:
        """Compact agent conversation history."""
        if runner:
            summary = await runner.compact()
            return {"summary": summary}
        return {"summary": ""}

    @router.on_event("shutdown")
    async def close_event_bus() -> None:
        if event_bus_publisher:
            event_bus_publisher.close()
            logger.info(
                "ipc.event_bus_publisher_closed",
                extra={"address": event_bus_publisher.address, "context": "api"},
            )

    async def _run_agent(runner: AgentRunner, message: str) -> None:
        """Run agent and consume all events."""
        async for _ in runner.run(message):
            pass  # Events are dispatched to consumers

    return router


def create_app(config: "VibeConfig", **kwargs: Any) -> Any:
    """Create a complete FastAPI application for the agent.

    Args:
        config: Vibe configuration
        **kwargs: Additional arguments passed to create_agent_router

    Returns:
        FastAPI application instance
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "FastAPI is not installed. Install with: pip install fastapi uvicorn"
        )

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(
        title="Vibe Agent API",
        description="Web API for Vibe AI agent interaction",
        version="1.0.0",
    )

    # Add CORS middleware for browser access
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add agent router
    router = create_agent_router(config, **kwargs)
    app.include_router(router, prefix="/api/agent", tags=["agent"])

    @app.get("/")
    async def root():
        return {"message": "Vibe Agent API", "docs": "/docs"}

    return app
