from typing import Dict, List, Optional, Any, Callable, Type
import asyncio
import logging
import time
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for different handling strategies"""

    NETWORK = "network"
    API_LIMIT = "api_limit"
    AUTHENTICATION = "authentication"
    VALIDATION = "validation"
    RESOURCE = "resource"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    UNKNOWN = "unknown"


@dataclass
class RetryConfig:
    """Configuration for retry behavior"""

    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    backoff_strategy: str = "exponential"  # exponential, linear, fixed


@dataclass
class ErrorDetails:
    """Detailed error information"""

    error_code: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    is_retryable: bool
    suggested_action: str
    technical_details: Optional[Dict] = None
    user_message: str = ""


class CircuitBreaker:
    """Circuit breaker pattern implementation"""

    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        if self.state == "OPEN":
            if (
                self.last_failure_time
                and time.time() - self.last_failure_time > self.recovery_timeout
            ):
                self.state = "HALF_OPEN"
            else:
                raise Exception(
                    "Circuit breaker is OPEN - service temporarily unavailable"
                )

        try:
            result = func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()

        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def reset(self):
        """Reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"


class ErrorHandler:
    """Comprehensive error handling and retry system"""

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.circuit_breakers = {}
        self.retry_configs = {
            ErrorCategory.NETWORK: RetryConfig(max_attempts=3, base_delay=2.0),
            ErrorCategory.API_LIMIT: RetryConfig(
                max_attempts=5, base_delay=5.0, max_delay=120.0
            ),
            ErrorCategory.TIMEOUT: RetryConfig(max_attempts=2, base_delay=1.0),
            ErrorCategory.SERVICE_UNAVAILABLE: RetryConfig(
                max_attempts=3, base_delay=10.0
            ),
        }
        self.error_history = []

    def _initialize_error_patterns(self) -> Dict[str, ErrorDetails]:
        """Initialize common error patterns"""
        return {
            "openai_rate_limit": ErrorDetails(
                error_code="OPENAI_RATE_LIMIT",
                message="OpenAI API rate limit exceeded",
                category=ErrorCategory.API_LIMIT,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                suggested_action="Automatically retrying with exponential backoff",
                user_message="Processing temporarily slowed due to high demand. Please wait...",
            ),
            "openai_timeout": ErrorDetails(
                error_code="OPENAI_TIMEOUT",
                message="OpenAI API request timeout",
                category=ErrorCategory.TIMEOUT,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                suggested_action="Retrying with shorter timeout",
                user_message="Request timed out. Retrying...",
            ),
            "chatterbox_unavailable": ErrorDetails(
                error_code="CHATTERBOX_UNAVAILABLE",
                message="Chatterbox TTS service unavailable",
                category=ErrorCategory.SERVICE_UNAVAILABLE,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                suggested_action="Checking service status and retrying",
                user_message="Voice generation service temporarily unavailable. Retrying...",
            ),
            "network_error": ErrorDetails(
                error_code="NETWORK_ERROR",
                message="Network connectivity issue",
                category=ErrorCategory.NETWORK,
                severity=ErrorSeverity.MEDIUM,
                is_retryable=True,
                suggested_action="Retrying with network backoff",
                user_message="Network issue detected. Retrying connection...",
            ),
            "validation_error": ErrorDetails(
                error_code="VALIDATION_ERROR",
                message="Input validation failed",
                category=ErrorCategory.VALIDATION,
                severity=ErrorSeverity.LOW,
                is_retryable=False,
                suggested_action="Check input parameters",
                user_message="Please check your input and try again",
            ),
            "resource_exhausted": ErrorDetails(
                error_code="RESOURCE_EXHAUSTED",
                message="System resources exhausted",
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                is_retryable=True,
                suggested_action="Waiting for resources to free up",
                user_message="System is busy. Please wait while we process your request...",
            ),
        }

    def classify_error(
        self, error: Exception, context: Optional[Dict] = None
    ) -> ErrorDetails:
        """Classify an error and return detailed information"""
        error_str = str(error).lower()
        error_type = type(error).__name__

        # Check for specific patterns
        if "rate limit" in error_str or "429" in error_str:
            return self.error_patterns["openai_rate_limit"]
        elif "timeout" in error_str or "timed out" in error_str:
            return self.error_patterns["openai_timeout"]
        elif "connection" in error_str or "network" in error_str:
            return self.error_patterns["network_error"]
        elif "validation" in error_str or "invalid" in error_str:
            return self.error_patterns["validation_error"]
        elif "chatterbox" in error_str and "unavailable" in error_str:
            return self.error_patterns["chatterbox_unavailable"]
        elif "memory" in error_str or "resource" in error_str:
            return self.error_patterns["resource_exhausted"]

        # Default unknown error
        return ErrorDetails(
            error_code="UNKNOWN_ERROR",
            message=str(error),
            category=ErrorCategory.UNKNOWN,
            severity=ErrorSeverity.MEDIUM,
            is_retryable=True,
            suggested_action="Attempting generic retry",
            user_message="An unexpected error occurred. Retrying...",
            technical_details={"error_type": error_type, "context": context},
        )

    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        context: Optional[Dict] = None,
        progress_callback: Optional[Callable] = None,
        **kwargs,
    ) -> Any:
        """Execute function with automatic retry logic"""
        last_error = None
        last_error_details = None

        for attempt in range(1, (retry_config.max_attempts if retry_config else 3) + 1):
            try:
                if progress_callback and attempt > 1:
                    await progress_callback(
                        f"Retrying after error (attempt {attempt})",
                        retry_info={
                            "attempt": attempt,
                            "max_attempts": retry_config.max_attempts
                            if retry_config
                            else 3,
                            "last_error": last_error_details.user_message
                            if last_error_details
                            else "Unknown error",
                        },
                    )

                # Execute function
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # Success - log recovery if this was a retry
                if attempt > 1:
                    logger.info(f"Function succeeded after {attempt} attempts")
                    if progress_callback:
                        await progress_callback(
                            "Recovered from error, continuing...",
                            retry_info={"recovered": True, "attempts_used": attempt},
                        )

                return result

            except Exception as error:
                last_error = error
                last_error_details = self.classify_error(error, context)

                # Log error details
                self._log_error(error, last_error_details, attempt, context)

                # Check if error is retryable
                if not last_error_details.is_retryable:
                    logger.error(f"Non-retryable error: {last_error_details.message}")
                    raise error

                # Check if this was the last attempt
                max_attempts = retry_config.max_attempts if retry_config else 3
                if attempt >= max_attempts:
                    logger.error(f"All {max_attempts} retry attempts failed")
                    raise error

                # Calculate delay
                delay = self._calculate_delay(
                    last_error_details.category, attempt, retry_config
                )

                logger.info(
                    f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})"
                )
                await asyncio.sleep(delay)

        # This should never be reached, but just in case
        if last_error:
            raise last_error

    def _calculate_delay(
        self,
        category: ErrorCategory,
        attempt: int,
        retry_config: Optional[RetryConfig] = None,
    ) -> float:
        """Calculate delay before next retry"""
        config = retry_config or self.retry_configs.get(category, RetryConfig())

        if config.backoff_strategy == "exponential":
            delay = config.base_delay * (config.exponential_base ** (attempt - 1))
        elif config.backoff_strategy == "linear":
            delay = config.base_delay * attempt
        else:  # fixed
            delay = config.base_delay

        # Apply maximum delay limit
        delay = min(delay, config.max_delay)

        # Add jitter to prevent thundering herd
        if config.jitter:
            jitter_range = delay * 0.1
            delay += random.uniform(-jitter_range, jitter_range)

        return max(delay, 0.1)  # Minimum 0.1 second delay

    def _log_error(
        self,
        error: Exception,
        error_details: ErrorDetails,
        attempt: int,
        context: Optional[Dict],
    ):
        """Log error with appropriate level based on severity"""
        log_message = f"Attempt {attempt} failed: {error_details.message}"

        error_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "error_code": error_details.error_code,
            "message": error_details.message,
            "category": error_details.category.value,
            "severity": error_details.severity.value,
            "attempt": attempt,
            "context": context,
            "technical_details": str(error),
        }

        self.error_history.append(error_record)

        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def get_circuit_breaker(self, service_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service"""
        if service_name not in self.circuit_breakers:
            self.circuit_breakers[service_name] = CircuitBreaker()
        return self.circuit_breakers[service_name]

    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified time period"""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        recent_errors = [
            error
            for error in self.error_history
            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
        ]

        summary = {
            "total_errors": len(recent_errors),
            "by_category": {},
            "by_severity": {},
            "by_error_code": {},
            "recent_errors": recent_errors[-10:],  # Last 10 errors
        }

        for error in recent_errors:
            # Count by category
            category = error["category"]
            summary["by_category"][category] = (
                summary["by_category"].get(category, 0) + 1
            )

            # Count by severity
            severity = error["severity"]
            summary["by_severity"][severity] = (
                summary["by_severity"].get(severity, 0) + 1
            )

            # Count by error code
            code = error["error_code"]
            summary["by_error_code"][code] = summary["by_error_code"].get(code, 0) + 1

        return summary


def with_error_handling(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
):
    """Decorator for automatic error handling and retries"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            error_handler = ErrorHandler()

            # Apply circuit breaker if specified
            if circuit_breaker_name:
                breaker = error_handler.get_circuit_breaker(circuit_breaker_name)
                try:
                    return await error_handler.execute_with_retry(
                        lambda: breaker.call(func, *args, **kwargs),
                        retry_config=retry_config,
                        context={
                            "function": func.__name__,
                            "circuit_breaker": circuit_breaker_name,
                        },
                    )
                except Exception as e:
                    logger.error(f"Circuit breaker call failed: {e}")
                    raise
            else:
                return await error_handler.execute_with_retry(
                    func,
                    *args,
                    retry_config=retry_config,
                    context={"function": func.__name__},
                    **kwargs,
                )

        return wrapper

    return decorator


# Global error handler instance
error_handler = ErrorHandler()
