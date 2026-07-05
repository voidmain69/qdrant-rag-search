"""Error tracking (Sentry) and telemetry (Prometheus).

Both are opt-in/config-driven: Sentry activates only when SENTRY_DSN is set,
/metrics only when METRICS_ENABLED=true (default). Neither adds overhead when off.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

    from app.core.config import Settings

logger = logging.getLogger(__name__)


def init_sentry(settings: Settings) -> None:
    if not settings.sentry_dsn:
        return
    import sentry_sdk

    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        environment=settings.environment.value,
        traces_sample_rate=settings.sentry_traces_sample_rate,
        send_default_pii=False,
    )
    logger.info("Sentry error tracking enabled (environment=%s)", settings.environment.value)


def setup_metrics(app: FastAPI) -> None:
    """Prometheus HTTP metrics (latency, status codes, in-progress) at GET /metrics."""
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        excluded_handlers=["/health", "/ready", "/metrics"],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
