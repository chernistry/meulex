"""OpenTelemetry setup and instrumentation for Meulex."""

import logging
from typing import Optional

from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from prometheus_client import Counter, Histogram, start_http_server

from meulex.config.settings import Settings

logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "meulex_requests_total",
    "Total number of requests",
    ["endpoint", "method", "status", "provider"]
)

REQUEST_DURATION = Histogram(
    "meulex_request_duration_seconds",
    "Request duration in seconds",
    ["endpoint", "method"]
)

CACHE_HITS = Counter(
    "meulex_cache_hits_total",
    "Total cache hits",
    ["type", "result"]
)

EMBEDDINGS_GENERATED = Counter(
    "meulex_embeddings_generated_total",
    "Total embeddings generated",
    ["model", "provider"]
)

DOCUMENTS_RETRIEVED = Counter(
    "meulex_documents_retrieved_total",
    "Total documents retrieved",
    ["strategy"]
)

LLM_TOKENS = Counter(
    "meulex_llm_tokens_total",
    "Total LLM tokens used",
    ["provider", "type"]
)

LLM_COST = Counter(
    "meulex_llm_cost_cents_total",
    "Total LLM cost in cents",
    ["provider"]
)

LLM_FAILURES = Counter(
    "meulex_llm_failures_total",
    "Total LLM failures",
    ["provider", "error_type"]
)

CIRCUIT_BREAKER_STATE = Counter(
    "meulex_circuit_breaker_state_total",
    "Circuit breaker state changes",
    ["component", "state"]
)

RAG_REQUESTS = Counter(
    "meulex_rag_requests_total",
    "Total RAG requests",
    ["status", "provider"]
)

RAG_REQUEST_DURATION = Histogram(
    "meulex_rag_request_duration_seconds",
    "RAG request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)

# Slack metrics
SLACK_EVENTS = Counter(
    "meulex_slack_events_total",
    "Total number of Slack events processed",
    ["event_type", "status"]
)


def setup_tracing(
    service_name: str = "meulex",
    version: str = "0.1.0",
    sample_rate: float = 1.0,
    console_export: bool = False
) -> None:
    """Set up OpenTelemetry tracing.
    
    Args:
        service_name: Name of the service
        version: Version of the service
        sample_rate: Sampling rate for traces (0.0 to 1.0)
        console_export: Whether to export traces to console
    """
    try:
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": version,
        })
        
        # Set up tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)
        
        # Add span processors
        if console_export:
            console_exporter = ConsoleSpanExporter()
            console_processor = BatchSpanProcessor(console_exporter)
            tracer_provider.add_span_processor(console_processor)
        
        logger.info(
            "OpenTelemetry tracing initialized",
            extra={
                "service_name": service_name,
                "version": version,
                "sample_rate": sample_rate
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to setup tracing: {e}")


def setup_metrics(
    service_name: str = "meulex",
    version: str = "0.1.0",
    prometheus_port: Optional[int] = None
) -> None:
    """Set up OpenTelemetry metrics.
    
    Args:
        service_name: Name of the service
        version: Version of the service
        prometheus_port: Port for Prometheus metrics server
    """
    try:
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": version,
        })
        
        # Set up metrics with Prometheus reader
        prometheus_reader = PrometheusMetricReader()
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[prometheus_reader]
        )
        metrics.set_meter_provider(meter_provider)
        
        # Start Prometheus metrics server if port specified
        if prometheus_port:
            start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        logger.info(
            "OpenTelemetry metrics initialized",
            extra={
                "service_name": service_name,
                "version": version,
                "prometheus_port": prometheus_port
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to setup metrics: {e}")


def instrument_fastapi(app) -> None:
    """Instrument FastAPI application with OpenTelemetry.
    
    Args:
        app: FastAPI application instance
    """
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def instrument_httpx() -> None:
    """Instrument HTTPX client with OpenTelemetry."""
    try:
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to instrument HTTPX: {e}")


def setup_observability(settings: Settings) -> None:
    """Set up complete observability stack.
    
    Args:
        settings: Application settings
    """
    if settings.tracing_enabled:
        setup_tracing(
            service_name=settings.otel_service_name,
            version=settings.otel_service_version,
            console_export=settings.debug
        )
        instrument_httpx()
    
    if settings.metrics_enabled:
        setup_metrics(
            service_name=settings.otel_service_name,
            version=settings.otel_service_version,
            prometheus_port=settings.prometheus_port if settings.debug else None
        )


def get_tracer(name: str) -> trace.Tracer:
    """Get a tracer instance.
    
    Args:
        name: Name of the tracer
        
    Returns:
        Tracer instance
    """
    return trace.get_tracer(name)


def get_meter(name: str) -> metrics.Meter:
    """Get a meter instance.
    
    Args:
        name: Name of the meter
        
    Returns:
        Meter instance
    """
    return metrics.get_meter(name)
