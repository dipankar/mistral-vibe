"""Resource management for planner: context budget, rate limiting, and timeouts."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum, auto
from typing import TYPE_CHECKING

from vibe.core.utils import logger

if TYPE_CHECKING:
    from vibe.core.config import PlannerConfig


class ResourceWarningLevel(StrEnum):
    """Warning levels for resource usage."""

    OK = auto()
    WARNING = auto()  # Approaching limit (>80%)
    CRITICAL = auto()  # Near limit (>95%)
    EXCEEDED = auto()  # Over limit


@dataclass
class TokenBudget:
    """Tracks token usage across plan execution."""

    max_tokens: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    warning_threshold: float = 0.80
    critical_threshold: float = 0.95

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.total_tokens)

    @property
    def utilization(self) -> float:
        if self.max_tokens <= 0:
            return 0.0
        return self.total_tokens / self.max_tokens

    @property
    def warning_level(self) -> ResourceWarningLevel:
        util = self.utilization
        if util >= 1.0:
            return ResourceWarningLevel.EXCEEDED
        if util >= self.critical_threshold:
            return ResourceWarningLevel.CRITICAL
        if util >= self.warning_threshold:
            return ResourceWarningLevel.WARNING
        return ResourceWarningLevel.OK

    def add_usage(self, prompt: int, completion: int) -> None:
        """Add token usage from a subagent execution."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion

    def can_proceed(self, estimated_tokens: int = 0) -> bool:
        """Check if we can proceed with estimated additional tokens."""
        return (self.total_tokens + estimated_tokens) < self.max_tokens

    def summary(self) -> str:
        """Return a human-readable summary."""
        return (
            f"{self.total_tokens:,}/{self.max_tokens:,} tokens "
            f"({self.utilization:.1%}) - {self.warning_level.value}"
        )


@dataclass
class RateLimitState:
    """Tracks rate limit errors and manages backoff."""

    max_retries: int = 5
    base_delay: float = 2.0  # seconds
    max_delay: float = 60.0  # seconds
    jitter_factor: float = 0.1

    # State
    consecutive_errors: int = 0
    total_errors: int = 0
    last_error_time: datetime | None = None
    current_delay: float = 0.0

    def record_error(self) -> float:
        """Record a rate limit error and return the delay to wait.

        Returns:
            The number of seconds to wait before retrying.
        """
        self.consecutive_errors += 1
        self.total_errors += 1
        self.last_error_time = datetime.utcnow()

        # Exponential backoff with jitter
        delay = min(
            self.base_delay * (2 ** (self.consecutive_errors - 1)),
            self.max_delay,
        )

        # Add jitter
        import random
        jitter = delay * self.jitter_factor * random.random()
        delay += jitter

        self.current_delay = delay
        return delay

    def record_success(self) -> None:
        """Record a successful request, resetting consecutive errors."""
        self.consecutive_errors = 0
        self.current_delay = 0.0

    def can_retry(self) -> bool:
        """Check if we can retry (haven't exceeded max retries)."""
        return self.consecutive_errors < self.max_retries

    @property
    def is_rate_limited(self) -> bool:
        """Check if we're currently in a rate-limited state."""
        return self.consecutive_errors > 0

    def summary(self) -> str:
        """Return a human-readable summary."""
        if not self.is_rate_limited:
            return "OK"
        return (
            f"Rate limited: {self.consecutive_errors}/{self.max_retries} retries, "
            f"waiting {self.current_delay:.1f}s"
        )


@dataclass
class DecisionTimeout:
    """Tracks timeout for pending decisions."""

    decision_id: str
    created_at: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: int = 0  # 0 = no timeout
    warning_issued: bool = False

    @property
    def elapsed_seconds(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds()

    @property
    def is_expired(self) -> bool:
        if self.timeout_seconds <= 0:
            return False
        return self.elapsed_seconds >= self.timeout_seconds

    @property
    def remaining_seconds(self) -> float:
        if self.timeout_seconds <= 0:
            return float("inf")
        return max(0, self.timeout_seconds - self.elapsed_seconds)

    def should_warn(self, warn_at_percent: float = 0.75) -> bool:
        """Check if we should issue a warning (75% of timeout elapsed)."""
        if self.timeout_seconds <= 0 or self.warning_issued:
            return False
        threshold = self.timeout_seconds * warn_at_percent
        return self.elapsed_seconds >= threshold


@dataclass
class CompactionRequest:
    """Request to compact context to free up token budget."""

    reason: str
    current_tokens: int
    max_tokens: int
    target_tokens: int  # Target after compaction
    urgency: ResourceWarningLevel

    @property
    def tokens_to_free(self) -> int:
        return max(0, self.current_tokens - self.target_tokens)


@dataclass
class PlannerResourceManager:
    """Centralized resource management for planner execution."""

    config: PlannerConfig
    token_budget: TokenBudget = field(init=False)
    rate_limit: RateLimitState = field(default_factory=RateLimitState)
    decision_timeouts: dict[str, DecisionTimeout] = field(default_factory=dict)
    _created_at: datetime = field(default_factory=datetime.utcnow)
    _compaction_requested: bool = field(default=False)
    _last_compaction_at: datetime | None = field(default=None)

    def __post_init__(self) -> None:
        self.token_budget = TokenBudget(max_tokens=self.config.max_context_tokens)

    # Token budget methods
    def add_token_usage(self, prompt: int, completion: int) -> ResourceWarningLevel:
        """Add token usage and return the new warning level."""
        self.token_budget.add_usage(prompt, completion)
        level = self.token_budget.warning_level
        if level in (ResourceWarningLevel.WARNING, ResourceWarningLevel.CRITICAL):
            logger.warning(
                "Planner context budget %s: %s",
                level.value,
                self.token_budget.summary(),
            )
        return level

    def can_start_step(self, estimated_tokens: int = 10000) -> bool:
        """Check if we can start a new step with estimated token usage."""
        return self.token_budget.can_proceed(estimated_tokens)

    def get_budget_summary(self) -> str:
        """Get a summary of the token budget."""
        return self.token_budget.summary()

    # Rate limit methods
    async def handle_rate_limit(self) -> bool:
        """Handle a rate limit error with backoff.

        Returns:
            True if we should retry, False if we've exceeded max retries.
        """
        if not self.rate_limit.can_retry():
            logger.error(
                "Rate limit retry budget exhausted after %d attempts",
                self.rate_limit.total_errors,
            )
            return False

        delay = self.rate_limit.record_error()
        logger.warning(
            "Rate limit hit, waiting %.1fs before retry (%d/%d)",
            delay,
            self.rate_limit.consecutive_errors,
            self.rate_limit.max_retries,
        )
        await asyncio.sleep(delay)
        return True

    def record_success(self) -> None:
        """Record a successful operation (resets rate limit state)."""
        self.rate_limit.record_success()

    def is_rate_limited(self) -> bool:
        """Check if we're currently rate limited."""
        return self.rate_limit.is_rate_limited

    def get_rate_limit_summary(self) -> str:
        """Get a summary of rate limit state."""
        return self.rate_limit.summary()

    # Decision timeout methods
    def start_decision_timeout(self, decision_id: str) -> None:
        """Start tracking timeout for a decision."""
        self.decision_timeouts[decision_id] = DecisionTimeout(
            decision_id=decision_id,
            timeout_seconds=self.config.decision_timeout_seconds,
        )

    def resolve_decision(self, decision_id: str) -> None:
        """Mark a decision as resolved (stop tracking timeout)."""
        self.decision_timeouts.pop(decision_id, None)

    def get_expired_decisions(self) -> list[str]:
        """Get list of decision IDs that have expired."""
        return [
            dt.decision_id
            for dt in self.decision_timeouts.values()
            if dt.is_expired
        ]

    def get_decisions_needing_warning(self) -> list[str]:
        """Get list of decision IDs that should receive a warning."""
        warnings = []
        for dt in self.decision_timeouts.values():
            if dt.should_warn():
                dt.warning_issued = True
                warnings.append(dt.decision_id)
        return warnings

    def get_decision_remaining_time(self, decision_id: str) -> float | None:
        """Get remaining time for a decision (None if not tracked)."""
        dt = self.decision_timeouts.get(decision_id)
        if dt:
            return dt.remaining_seconds
        return None

    # Auto-compaction methods
    def check_compaction_needed(self) -> CompactionRequest | None:
        """Check if context compaction is needed based on token budget.

        Returns:
            CompactionRequest if compaction should be triggered, None otherwise.
        """
        level = self.token_budget.warning_level

        # Only trigger compaction at critical or exceeded levels
        if level not in (ResourceWarningLevel.CRITICAL, ResourceWarningLevel.EXCEEDED):
            return None

        # Don't request compaction too frequently (minimum 30 seconds between requests)
        if self._last_compaction_at:
            elapsed = (datetime.utcnow() - self._last_compaction_at).total_seconds()
            if elapsed < 30:
                return None

        # Calculate target tokens (aim for 70% utilization after compaction)
        target_ratio = 0.70
        target_tokens = int(self.token_budget.max_tokens * target_ratio)

        return CompactionRequest(
            reason=f"Context budget {level.value}: {self.token_budget.utilization:.1%} utilized",
            current_tokens=self.token_budget.total_tokens,
            max_tokens=self.token_budget.max_tokens,
            target_tokens=target_tokens,
            urgency=level,
        )

    def request_compaction(self) -> CompactionRequest | None:
        """Request compaction and mark it as requested.

        Returns:
            CompactionRequest if compaction is needed, None if already requested or not needed.
        """
        if self._compaction_requested:
            return None

        request = self.check_compaction_needed()
        if request:
            self._compaction_requested = True
            logger.info(
                "Compaction requested: %s (need to free %d tokens)",
                request.reason,
                request.tokens_to_free,
            )
        return request

    def record_compaction_complete(self, tokens_freed: int) -> None:
        """Record that compaction has completed.

        Args:
            tokens_freed: Number of tokens freed by compaction.
        """
        self._compaction_requested = False
        self._last_compaction_at = datetime.utcnow()

        # Adjust token budget to reflect freed tokens
        # Note: This is approximate - actual token count should be re-measured
        if tokens_freed > 0:
            # Reduce the recorded usage proportionally
            reduction_ratio = tokens_freed / max(1, self.token_budget.total_tokens)
            self.token_budget.prompt_tokens = int(
                self.token_budget.prompt_tokens * (1 - reduction_ratio)
            )
            self.token_budget.completion_tokens = int(
                self.token_budget.completion_tokens * (1 - reduction_ratio)
            )

        logger.info(
            "Compaction complete: freed ~%d tokens, now at %s",
            tokens_freed,
            self.token_budget.summary(),
        )

    @property
    def compaction_pending(self) -> bool:
        """Check if a compaction request is pending."""
        return self._compaction_requested

    # Overall status
    def get_status_summary(self) -> dict[str, str]:
        """Get overall resource status summary."""
        return {
            "tokens": self.get_budget_summary(),
            "rate_limit": self.get_rate_limit_summary(),
            "pending_decisions": str(len(self.decision_timeouts)),
            "uptime": f"{(datetime.utcnow() - self._created_at).total_seconds():.0f}s",
        }

    def should_pause_for_resources(self) -> tuple[bool, str]:
        """Check if we should pause execution due to resource constraints.

        Returns:
            Tuple of (should_pause, reason).
        """
        # Check token budget
        if self.token_budget.warning_level == ResourceWarningLevel.EXCEEDED:
            return True, "Token budget exceeded"

        # Check rate limit
        if not self.rate_limit.can_retry():
            return True, "Rate limit retry budget exhausted"

        # Check expired decisions
        expired = self.get_expired_decisions()
        if expired:
            return True, f"Decision timeout expired: {', '.join(expired)}"

        return False, ""


# Standalone retry decorator for rate-limited operations
async def with_rate_limit_retry(
    operation,
    resource_manager: PlannerResourceManager,
    operation_name: str = "operation",
):
    """Execute an operation with rate limit retry handling.

    Args:
        operation: Async callable to execute
        resource_manager: Resource manager instance
        operation_name: Name for logging

    Returns:
        Result of the operation

    Raises:
        Exception: If operation fails after all retries
    """
    last_error = None

    while True:
        try:
            result = await operation()
            resource_manager.record_success()
            return result
        except Exception as exc:
            # Check if it's a rate limit error
            error_str = str(exc).lower()
            is_rate_limit = any(
                indicator in error_str
                for indicator in ["rate limit", "429", "too many requests", "quota"]
            )

            if not is_rate_limit:
                raise

            last_error = exc
            should_retry = await resource_manager.handle_rate_limit()
            if not should_retry:
                logger.error(
                    "Failed %s after rate limit retries: %s",
                    operation_name,
                    exc,
                )
                raise

    # Should not reach here, but just in case
    if last_error:
        raise last_error
