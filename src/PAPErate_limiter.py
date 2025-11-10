# --------------------------------
# Pick-a-Path AI Ethics
# Rate Limiting for API calls
# --------------------------------

import time
from collections import defaultdict, deque
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter for API calls.
    Prevents excessive API usage and cost overruns.
    """

    def __init__(
        self,
        max_requests_per_minute: int = 20,
        max_requests_per_hour: int = 100,
        max_tokens_per_minute: int = 40000,
        max_cost_per_day: float = 10.0  # USD
    ):
        """
        Initialise the rate limiter.

        Args:
            max_requests_per_minute: Maximum API requests per minute
            max_requests_per_hour: Maximum API requests per hour
            max_tokens_per_minute: Maximum tokens per minute
            max_cost_per_day: Maximum cost per day in USD
        """
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.max_tokens_per_minute = max_tokens_per_minute
        self.max_cost_per_day = max_cost_per_day

        # Track requests with timestamps
        self.request_times_minute = deque(maxlen=max_requests_per_minute)
        self.request_times_hour = deque(maxlen=max_requests_per_hour)

        # Track tokens
        self.token_usage_minute = deque(maxlen=100)  # Store last 100 token counts

        # Track daily cost
        self.daily_cost = defaultdict(float)  # date -> cost
        self.last_reset_date = time.strftime("%Y-%m-%d")

    def _cleanup_old_entries(self):
        """Remove timestamps older than tracking windows."""
        current_time = time.time()

        # Clean minute window
        while self.request_times_minute and current_time - self.request_times_minute[0] > 60:
            self.request_times_minute.popleft()

        # Clean hour window
        while self.request_times_hour and current_time - self.request_times_hour[0] > 3600:
            self.request_times_hour.popleft()

        # Clean token usage minute window
        while self.token_usage_minute and current_time - self.token_usage_minute[0][0] > 60:
            self.token_usage_minute.popleft()

        # Reset daily cost if new day
        current_date = time.strftime("%Y-%m-%d")
        if current_date != self.last_reset_date:
            logger.info("Resetting daily cost counter for new day: %s", current_date)
            self.daily_cost.clear()
            self.last_reset_date = current_date

    def check_rate_limit(self, estimated_tokens: int = 1000) -> tuple[bool, Optional[str]]:
        """
        Check if a request can be made within rate limits.

        Args:
            estimated_tokens: Estimated number of tokens for the request

        Returns:
            Tuple of (allowed, reason_if_blocked)
        """
        self._cleanup_old_entries()
        current_time = time.time()

        # Check requests per minute
        if len(self.request_times_minute) >= self.max_requests_per_minute:
            wait_time = 60 - (current_time - self.request_times_minute[0])
            return False, f"Rate limit exceeded: {self.max_requests_per_minute} requests/minute. Wait {wait_time:.1f}s"

        # Check requests per hour
        if len(self.request_times_hour) >= self.max_requests_per_hour:
            wait_time = 3600 - (current_time - self.request_times_hour[0])
            return False, f"Rate limit exceeded: {self.max_requests_per_hour} requests/hour. Wait {wait_time/60:.1f}m"

        # Check tokens per minute
        tokens_in_last_minute = sum(tokens for ts, tokens in self.token_usage_minute if current_time - ts < 60)
        if tokens_in_last_minute + estimated_tokens > self.max_tokens_per_minute:
            return False, f"Token rate limit exceeded: {self.max_tokens_per_minute} tokens/minute"

        # Check daily cost
        current_date = time.strftime("%Y-%m-%d")
        if self.daily_cost[current_date] >= self.max_cost_per_day:
            return False, f"Daily cost limit exceeded: ${self.max_cost_per_day:.2f}"

        return True, None

    def record_request(self, tokens_used: int, cost: float = 0.0):
        """
        Record a successful API request.

        Args:
            tokens_used: Number of tokens used in the request
            cost: Cost of the request in USD
        """
        current_time = time.time()
        current_date = time.strftime("%Y-%m-%d")

        self.request_times_minute.append(current_time)
        self.request_times_hour.append(current_time)
        self.token_usage_minute.append((current_time, tokens_used))
        self.daily_cost[current_date] += cost

        logger.debug(
            "Request recorded: %d tokens, $%.4f cost, daily total: $%.4f",
            tokens_used, cost, self.daily_cost[current_date]
        )

    def get_stats(self) -> dict:
        """
        Get current rate limiter statistics.

        Returns:
            Dictionary with current usage stats
        """
        self._cleanup_old_entries()
        current_time = time.time()
        current_date = time.strftime("%Y-%m-%d")

        tokens_in_last_minute = sum(
            tokens for ts, tokens in self.token_usage_minute if current_time - ts < 60
        )

        return {
            "requests_last_minute": len(self.request_times_minute),
            "requests_last_hour": len(self.request_times_hour),
            "tokens_last_minute": tokens_in_last_minute,
            "daily_cost": self.daily_cost[current_date],
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour,
            "max_tokens_per_minute": self.max_tokens_per_minute,
            "max_cost_per_day": self.max_cost_per_day,
        }


# Model pricing (as of 2024 - update as needed)
MODEL_PRICING = {
    "gpt-4o": {"input": 0.0025 / 1000, "output": 0.01 / 1000},  # per token
    "gpt-4o-mini": {"input": 0.00015 / 1000, "output": 0.0006 / 1000},
    "gpt-4.1": {"input": 0.003 / 1000, "output": 0.012 / 1000},
    "gpt-4.1-mini": {"input": 0.0002 / 1000, "output": 0.0008 / 1000},
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of an API call.

    Args:
        model: Model name (e.g., 'gpt-4o-mini')
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    if model not in MODEL_PRICING:
        logger.warning("Unknown model for cost estimation: %s, using gpt-4o-mini pricing", model)
        model = "gpt-4o-mini"

    pricing = MODEL_PRICING[model]
    cost = (input_tokens * pricing["input"]) + (output_tokens * pricing["output"])
    return cost
