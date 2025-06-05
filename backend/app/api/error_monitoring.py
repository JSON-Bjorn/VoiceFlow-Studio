from fastapi import APIRouter, Depends, HTTPException, Query
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from ..core.auth import get_current_user
from ..models.user import User
from ..services.error_handler import error_handler
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/status")
async def get_error_monitoring_status(
    hours: int = Query(default=24, ge=1, le=168),  # 1 hour to 1 week
    current_user: User = Depends(get_current_user),
):
    """
    Get comprehensive error monitoring status and statistics

    Args:
        hours: Time period to analyze (1-168 hours)

    Returns:
        Detailed error monitoring information including:
        - Error summary by category and severity
        - Circuit breaker status
        - Retry statistics
        - Recent error patterns
    """
    try:
        # Get error summary
        error_summary = error_handler.get_error_summary(hours=hours)

        # Get circuit breaker status
        circuit_breaker_status = {}
        for service_name, breaker in error_handler.circuit_breakers.items():
            circuit_breaker_status[service_name] = {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
                "last_failure_time": datetime.fromtimestamp(
                    breaker.last_failure_time
                ).isoformat()
                if breaker.last_failure_time
                else None,
                "failure_threshold": breaker.failure_threshold,
                "recovery_timeout": breaker.recovery_timeout,
            }

        # Calculate error trends
        error_trends = _calculate_error_trends(error_summary, hours)

        # Get retry configuration
        retry_configs = {}
        for category, config in error_handler.retry_configs.items():
            retry_configs[category.value] = {
                "max_attempts": config.max_attempts,
                "base_delay": config.base_delay,
                "max_delay": config.max_delay,
                "exponential_base": config.exponential_base,
                "backoff_strategy": config.backoff_strategy,
            }

        return {
            "success": True,
            "monitoring_period_hours": hours,
            "timestamp": datetime.utcnow().isoformat(),
            "error_summary": error_summary,
            "circuit_breakers": circuit_breaker_status,
            "error_trends": error_trends,
            "retry_configurations": retry_configs,
            "system_health": _calculate_system_health(
                error_summary, circuit_breaker_status
            ),
            "recommendations": _generate_recommendations(
                error_summary, circuit_breaker_status
            ),
        }

    except Exception as e:
        logger.error(f"Error monitoring status failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get error monitoring status: {str(e)}"
        )


@router.get("/patterns")
async def get_error_patterns(
    category: Optional[str] = Query(default=None),
    severity: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    """
    Get detailed error patterns and analysis

    Args:
        category: Filter by error category (optional)
        severity: Filter by error severity (optional)
        limit: Maximum number of errors to return

    Returns:
        Detailed error pattern analysis
    """
    try:
        # Filter errors based on criteria
        filtered_errors = error_handler.error_history[-limit:]

        if category:
            filtered_errors = [
                e for e in filtered_errors if e.get("category") == category
            ]

        if severity:
            filtered_errors = [
                e for e in filtered_errors if e.get("severity") == severity
            ]

        # Analyze patterns
        patterns = _analyze_error_patterns(filtered_errors)

        return {
            "success": True,
            "total_errors_analyzed": len(filtered_errors),
            "filters_applied": {
                "category": category,
                "severity": severity,
                "limit": limit,
            },
            "patterns": patterns,
            "error_details": filtered_errors,
        }

    except Exception as e:
        logger.error(f"Error pattern analysis failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to analyze error patterns: {str(e)}"
        )


@router.post("/reset-circuit-breaker/{service_name}")
async def reset_circuit_breaker(
    service_name: str, current_user: User = Depends(get_current_user)
):
    """
    Manually reset a circuit breaker for a specific service

    Args:
        service_name: Name of the service circuit breaker to reset

    Returns:
        Result of circuit breaker reset
    """
    try:
        if service_name not in error_handler.circuit_breakers:
            raise HTTPException(
                status_code=404,
                detail=f"Circuit breaker for service '{service_name}' not found",
            )

        breaker = error_handler.circuit_breakers[service_name]
        old_state = breaker.state
        old_failure_count = breaker.failure_count

        breaker.reset()

        logger.info(
            f"Circuit breaker for '{service_name}' manually reset by user {current_user.id}"
        )

        return {
            "success": True,
            "service_name": service_name,
            "message": f"Circuit breaker for '{service_name}' has been reset",
            "previous_state": {"state": old_state, "failure_count": old_failure_count},
            "current_state": {
                "state": breaker.state,
                "failure_count": breaker.failure_count,
            },
            "reset_by": current_user.email,
            "reset_at": datetime.utcnow().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Circuit breaker reset failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to reset circuit breaker: {str(e)}"
        )


@router.get("/health")
async def get_system_health():
    """
    Get overall system health status (public endpoint)

    Returns:
        High-level system health information
    """
    try:
        error_summary = error_handler.get_error_summary(hours=1)  # Last hour

        # Calculate health metrics
        recent_errors = error_summary["total_errors"]
        critical_errors = error_summary["by_severity"].get("critical", 0)
        high_errors = error_summary["by_severity"].get("high", 0)

        # Determine health status
        if critical_errors > 0:
            health_status = "critical"
            health_score = 0
        elif high_errors > 5:
            health_status = "degraded"
            health_score = 30
        elif recent_errors > 20:
            health_status = "unstable"
            health_score = 60
        elif recent_errors > 5:
            health_status = "warning"
            health_score = 80
        else:
            health_status = "healthy"
            health_score = 100

        # Check circuit breaker status
        open_breakers = sum(
            1
            for breaker in error_handler.circuit_breakers.values()
            if breaker.state == "OPEN"
        )

        return {
            "success": True,
            "health_status": health_status,
            "health_score": health_score,
            "last_hour_errors": recent_errors,
            "critical_errors": critical_errors,
            "high_severity_errors": high_errors,
            "open_circuit_breakers": open_breakers,
            "total_circuit_breakers": len(error_handler.circuit_breakers),
            "timestamp": datetime.utcnow().isoformat(),
            "uptime_status": "operational",  # Could be enhanced with actual uptime monitoring
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "success": False,
            "health_status": "unknown",
            "health_score": 0,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


def _calculate_error_trends(error_summary: Dict, hours: int) -> Dict[str, Any]:
    """Calculate error trends over time"""

    # For a simple implementation, we'll calculate basic trends
    # In a production system, you'd want more sophisticated time-series analysis

    total_errors = error_summary["total_errors"]
    errors_per_hour = total_errors / hours if hours > 0 else 0

    # Determine trend based on error distribution
    if total_errors == 0:
        trend = "stable"
        trend_direction = "neutral"
    elif errors_per_hour > 5:
        trend = "increasing"
        trend_direction = "up"
    elif errors_per_hour > 2:
        trend = "moderate"
        trend_direction = "up"
    else:
        trend = "stable"
        trend_direction = "neutral"

    return {
        "trend": trend,
        "trend_direction": trend_direction,
        "errors_per_hour": round(errors_per_hour, 2),
        "total_errors": total_errors,
        "period_hours": hours,
        "most_common_category": max(
            error_summary["by_category"].items(), key=lambda x: x[1]
        )[0]
        if error_summary["by_category"]
        else None,
        "most_common_severity": max(
            error_summary["by_severity"].items(), key=lambda x: x[1]
        )[0]
        if error_summary["by_severity"]
        else None,
    }


def _calculate_system_health(
    error_summary: Dict, circuit_breaker_status: Dict
) -> Dict[str, Any]:
    """Calculate overall system health metrics"""

    total_errors = error_summary["total_errors"]
    critical_errors = error_summary["by_severity"].get("critical", 0)
    high_errors = error_summary["by_severity"].get("high", 0)
    open_breakers = sum(
        1 for status in circuit_breaker_status.values() if status["state"] == "OPEN"
    )

    # Calculate health score (0-100)
    health_score = 100

    # Deduct points for errors
    health_score -= critical_errors * 20  # Critical errors are severe
    health_score -= high_errors * 10  # High errors are significant
    health_score -= (total_errors - critical_errors - high_errors) * 2  # Other errors
    health_score -= open_breakers * 15  # Open circuit breakers

    health_score = max(0, health_score)  # Don't go below 0

    # Determine status
    if health_score >= 90:
        status = "excellent"
    elif health_score >= 75:
        status = "good"
    elif health_score >= 50:
        status = "fair"
    elif health_score >= 25:
        status = "poor"
    else:
        status = "critical"

    return {
        "overall_score": health_score,
        "status": status,
        "factors": {
            "total_errors": total_errors,
            "critical_errors": critical_errors,
            "high_errors": high_errors,
            "open_circuit_breakers": open_breakers,
        },
    }


def _generate_recommendations(
    error_summary: Dict, circuit_breaker_status: Dict
) -> List[str]:
    """Generate actionable recommendations based on error patterns"""

    recommendations = []

    total_errors = error_summary["total_errors"]
    critical_errors = error_summary["by_severity"].get("critical", 0)
    high_errors = error_summary["by_severity"].get("high", 0)

    # Error-based recommendations
    if critical_errors > 0:
        recommendations.append(
            "Immediate attention required: Critical errors detected. Check system logs and address underlying issues."
        )

    if high_errors > 10:
        recommendations.append(
            "High number of high-severity errors. Consider implementing additional error handling or increasing system resources."
        )

    if total_errors > 50:
        recommendations.append(
            "High error volume detected. Review recent deployments and system changes."
        )

    # Circuit breaker recommendations
    open_breakers = [
        name
        for name, status in circuit_breaker_status.items()
        if status["state"] == "OPEN"
    ]

    if open_breakers:
        recommendations.append(
            f"Circuit breakers open for: {', '.join(open_breakers)}. Check service health and consider manual reset if services are restored."
        )

    # Category-specific recommendations
    error_categories = error_summary["by_category"]

    if error_categories.get("network", 0) > 10:
        recommendations.append(
            "High number of network errors. Check network connectivity and external service availability."
        )

    if error_categories.get("api_limit", 0) > 5:
        recommendations.append(
            "API rate limiting detected. Consider implementing request throttling or upgrading API limits."
        )

    if error_categories.get("resource", 0) > 5:
        recommendations.append(
            "Resource exhaustion detected. Monitor system resources and consider scaling up."
        )

    # If no issues found
    if not recommendations:
        recommendations.append(
            "System is operating normally. Continue monitoring for any changes in error patterns."
        )

    return recommendations


def _analyze_error_patterns(errors: List[Dict]) -> Dict[str, Any]:
    """Analyze patterns in error data"""

    if not errors:
        return {"pattern_analysis": "No errors to analyze"}

    # Time-based patterns
    error_times = [datetime.fromisoformat(e["timestamp"]) for e in errors]
    if error_times:
        time_span = (
            max(error_times) - min(error_times)
        ).total_seconds() / 3600  # hours
        error_rate = len(errors) / max(time_span, 1)
    else:
        error_rate = 0

    # Context patterns
    contexts = [e.get("context", {}) for e in errors]
    function_failures = {}

    for context in contexts:
        if isinstance(context, dict) and "function" in context:
            func = context["function"]
            function_failures[func] = function_failures.get(func, 0) + 1

    # Most problematic functions
    top_failing_functions = sorted(
        function_failures.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return {
        "error_rate_per_hour": round(error_rate, 2),
        "time_span_hours": round(time_span, 2) if "time_span" in locals() else 0,
        "top_failing_functions": top_failing_functions,
        "pattern_analysis": "Completed",
        "recommendations": [
            f"Function '{func}' failed {count} times - investigate implementation"
            for func, count in top_failing_functions[:3]
            if count > 3
        ],
    }
