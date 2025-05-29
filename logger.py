# #Copyright (C) 2025 β ORI Inc.
# #Written by Awase Khirni Syed 2025

'''
Logger class with W3c Trace Context headers support, span ID and Trace Parentt fields, OpenTelemtry-compliant structured logging.
Added component lifecycle monitoring, full context propogation, async-safe operation, layered architectural visibility
added structured machine-readable logs.  This will help ins designing power dashboard, alerts and root cause analysis
to do
-- think about way to include logging to measure end of life and technical debt here

'''
import logging
import os
import json
import uuid
import time
import asyncio
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from logging.handlers import RotatingFileHandler
from pythonjsonlogger import jsonlogger
from dotenv import load_dotenv
from flask import request, g, has_request_context
from typing import Optional, Dict, Any, Callable, Union, List

load_dotenv()

# Constants
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Custom Log Levels
AUDIT_LEVEL_NUM = 25
SECURITY_LEVEL_NUM = 35
LIFECYCLE_LEVEL_NUM = 15
TRACE_LEVEL_NUM = 9
logging.addLevelName(AUDIT_LEVEL_NUM, "AUDIT")
logging.addLevelName(SECURITY_LEVEL_NUM, "SECURITY")
logging.addLevelName(LIFECYCLE_LEVEL_NUM, "LIFECYCLE")
logging.addLevelName(TRACE_LEVEL_NUM, "TRACE")

def lifecycle(self, message, *args, **kwargs):
    if self.isEnabledFor(LIFECYCLE_LEVEL_NUM):
        self._log(LIFECYCLE_LEVEL_NUM, message, args, **kwargs)

def trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE_LEVEL_NUM):
        self._log(TRACE_LEVEL_NUM, message, args, **kwargs)

logging.Logger.lifecycle = lifecycle
logging.Logger.trace = trace

class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Enhanced JSON formatter with W3C Trace Context, OpenTelemetry compatibility."""

    def add_fields(self, log_record: Dict, record: logging.LogRecord, message_dict: Dict) -> None:
        super().add_fields(log_record, record, message_dict)

        # Base metadata
        log_record.update({
            "level": record.levelname,
            "timestamp": self.formatTime(record, self.datefmt),
            "logger_name": record.name,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "module": record.module,
            "function": record.funcName,
            "process_id": record.process,
            "thread_id": record.thread,
            "service": os.getenv("SERVICE_NAME", "unknown"),
            "environment": os.getenv("ENVIRONMENT", "development"),
        })

        # Request context
        if hasattr(record, 'context'):
            log_record.update(record.context)

        # Exception details
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Message field handling
        if message_dict:
            log_record.update(message_dict)
        elif isinstance(record.msg, dict):
            log_record.update(record.msg)
        else:
            log_record["message"] = record.getMessage()

        # Ensure OTel/W3C compatible fields
        log_record["trace_id"] = log_record.get("trace_id", "N/A")
        log_record["span_id"] = log_record.get("span_id", "N/A")
        log_record["trace_flags"] = log_record.get("trace_flags", "01")  # Sampled by default

class AsyncLogHandler:
    """Wrapper for async logging operations with distributed tracing context propagation."""

    _executor = ThreadPoolExecutor(max_workers=5)

    @classmethod
    async def log_async(cls, logger: logging.Logger, level: int, message: str, **context: Any) -> None:
        """Execute logging in a separate thread."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            cls._executor,
            lambda: logger.log(level, message, extra={"context": context})
        )

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger with file handlers and W3C trace context support."""

    logger = logging.getLogger(name)
    logger.setLevel(os.getenv("LOG_LEVEL", "DEBUG").upper())

    json_formatter = CustomJsonFormatter(
        fmt="%(timestamp)s %(level)s %(logger_name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z"
    )

    def create_handler(filename: str, level: int) -> RotatingFileHandler:
        handler = RotatingFileHandler(
            os.path.join(LOG_DIR, filename),
            maxBytes=5_000_000,
            backupCount=3,
            encoding='utf-8'
        )
        handler.setLevel(level)
        handler.setFormatter(json_formatter)
        return handler

    if not logger.handlers:
        logger.addHandler(create_handler("app.log", logging.DEBUG))
        logger.addHandler(create_handler("error.log", logging.ERROR))
        logger.addHandler(create_handler("audit.log", AUDIT_LEVEL_NUM))
        logger.addHandler(create_handler("security.log", SECURITY_LEVEL_NUM))
        logger.addHandler(create_handler("lifecycle.log", LIFECYCLE_LEVEL_NUM))
        logger.addHandler(create_handler("trace.log", TRACE_LEVEL_NUM))

    logger.propagate = False
    return logger

def init_request_context() -> Dict[str, str]:
    """Initialize request context with IDs for tracing including W3C Trace Context."""
    traceparent = request.headers.get('traceparent', f"00-{str(uuid.uuid4())}-{str(uuid.uuid4())[:16]}-01")

    try:
        _, trace_id, span_id, trace_flags = traceparent.split('-')
    except ValueError:
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]
        trace_flags = "01"

    context = {
        'request_id': str(uuid.uuid4()),
        'correlation_id': request.headers.get('X-Correlation-ID', str(uuid.uuid4())),
        'trace_id': trace_id,
        'span_id': span_id,
        'trace_flags': trace_flags
    }

    g.update(context)
    return context

def get_request_context() -> Dict[str, str]:
    """Get current request context safely"""
    if has_request_context():
        return {
            'request_id': getattr(g, 'request_id', 'N/A'),
            'correlation_id': getattr(g, 'correlation_id', 'N/A'),
            'trace_id': getattr(g, 'trace_id', 'N/A'),
            'span_id': getattr(g, 'span_id', 'N/A'),
            'trace_flags': getattr(g, 'trace_flags', 'N/A')
        }
    return {}

def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    layer: Optional[str] = None,
    span_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    **additional_context: Any
) -> None:
    """Log messages with full request context and layer tracing."""
    context = {
        **get_request_context(),
        **{
            "layer": layer or "unknown",
            "span_id": span_id or str(uuid.uuid4())[:16],
            "trace_id": trace_id or str(uuid.uuid4()),
            "trace_flags": "01",
            **additional_context
        }
    }

    logger.log(level, message, extra={'context': context})

# Decorator for lifecycle logging
def log_component_init(logger: logging.Logger):
    def decorator(cls):
        original_init = cls.__init__

        def __init__(self, *args, **kwargs):
            log_with_context(logger, LIFECYCLE_LEVEL_NUM, f"{cls.__name__} initialized", layer="lifecycle", action="initialized")
            original_init(self, *args, **kwargs)

        cls.__init__ = __init__
        return cls
    return decorator

# Decorator for function/method tracing with W3C context propagation
def log_execution(layer: str = "unknown"):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            ctx = get_request_context()
            span_id = str(uuid.uuid4())[:16]

            logger = logging.getLogger(func.__module__)
            trace_id = ctx.get("trace_id", str(uuid.uuid4()))

            log_with_context(
                logger,
                TRACE_LEVEL_NUM,
                f"Started {func.__qualname__}",
                layer=layer,
                span_id=span_id,
                trace_id=trace_id,
                args=str(args),
                kwargs=str(kwargs),
                action="start"
            )

            try:
                result = func(*args, **kwargs)
                duration = round((time.time() - start_time) * 1000, 2)
                log_with_context(
                    logger,
                    TRACE_LEVEL_NUM,
                    f"Finished {func.__qualname__}",
                    layer=layer,
                    span_id=span_id,
                    trace_id=trace_id,
                    duration_ms=duration,
                    result=str(result)[:200],
                    action="end"
                )
                return result
            except Exception as e:
                duration = round((time.time() - start_time) * 1000, 2)
                log_with_context(
                    logger,
                    logging.ERROR,
                    f"Failed {func.__qualname__}: {str(e)}",
                    layer=layer,
                    span_id=span_id,
                    trace_id=trace_id,
                    duration_ms=duration,
                    exc_info=True,
                    error=str(e),
                    traceback=str(e.__traceback__)
                )
                raise
        return wrapper
    return decorator

# Flask-specific integration
def before_request() -> None:
    """Initialize request logging context with W3C Trace Context support."""
    init_request_context()
