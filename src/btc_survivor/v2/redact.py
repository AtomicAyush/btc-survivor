"""Mask registered secret values in every text that leaves the process.

That covers log lines with their tracebacks, uncaught exceptions in the main thread, other threads and the asyncio
loop, errors Python can only ignore, such as one raised in __del__, recorded feed lines such as close reasons, and
stored details. Only local_secrets registers values.
"""

from __future__ import annotations

import logging
import sys
import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import asyncio

MASK = "[redacted]"
LOGGER = "btc_survivor.v2"

_values: set[str] = set()


def register(value: str) -> None:
    """Remember a secret value so that redact masks it from now on. An empty value would mask everything."""
    if value:
        _values.add(value)


def redact(text: str) -> str:
    """Return text with every registered value masked, longest first so overlapping values vanish whole."""
    for value in sorted(_values, key=len, reverse=True):
        if value in text:
            text = text.replace(value, MASK)
    return text


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and masks its whole output, so traceback text from formatException is masked too."""

    def __init__(self, inner: logging.Formatter | None = None) -> None:
        super().__init__()
        self.inner = inner or logging.Formatter()

    def format(self, record: logging.LogRecord) -> str:
        text = redact(self.inner.format(record))
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return text

    def formatException(self, ei: Any) -> str:
        return redact(self.inner.formatException(ei))

    def formatStack(self, stack_info: str) -> str:
        return redact(self.inner.formatStack(stack_info))


def _all_handlers() -> list[logging.Handler]:
    loggers = [logging.getLogger(), *(logger for logger in logging.root.manager.loggerDict.values()
                                      if isinstance(logger, logging.Logger))]
    return [handler for logger in loggers for handler in logger.handlers]


def install_formatter() -> None:
    """Wrap the formatter of every handler that exists now, and of the last resort handler that logging uses for a
    logger with no handler, adding a stderr handler first when the root has none.

    Call it after logging is set up, because a handler added later is not wrapped.
    """
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(logging.StreamHandler(sys.stderr))
    last = [logging.lastResort] if logging.lastResort is not None else []
    for handler in _all_handlers() + last:
        if not isinstance(handler.formatter, RedactingFormatter):
            handler.setFormatter(RedactingFormatter(handler.formatter))


def _excepthook(exc_type: type[BaseException], exc: BaseException, tb: Any) -> None:
    logging.getLogger(LOGGER).critical("uncaught exception", exc_info=(exc_type, exc, tb))


def _thread_excepthook(args: threading.ExceptHookArgs) -> None:
    name = args.thread.name if args.thread is not None else "unknown"
    logging.getLogger(LOGGER).critical("uncaught exception in thread %s", name,
                                       exc_info=(args.exc_type, args.exc_value, args.exc_traceback))


def _unraisablehook(unraisable: Any) -> None:
    message = unraisable.err_msg or "Exception ignored in"
    logging.getLogger(LOGGER).error("%s: %r", message, unraisable.object,
                                    exc_info=(unraisable.exc_type, unraisable.exc_value, unraisable.exc_traceback))


def install() -> None:
    """Route uncaught and ignored exceptions and warnings through logging and mask every handler, so stderr never
    sees a secret. A log call that fails to format is dropped, because logging would print its arguments unmasked."""
    logging.captureWarnings(True)
    logging.raiseExceptions = False
    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook
    sys.unraisablehook = _unraisablehook
    install_formatter()


def asyncio_handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
    """Log an asyncio error through logging, with the same context lines the default handler prints."""
    exc = context.get("exception")
    extra = "".join(f"\n{key}: {value!r}" for key, value in sorted(context.items())
                    if key not in ("message", "exception"))
    logging.getLogger(LOGGER).error("asyncio: %s%s", context.get("message", "unhandled error"), extra,
                                    exc_info=(type(exc), exc, exc.__traceback__) if exc is not None else None)


def install_asyncio_handler(loop: asyncio.AbstractEventLoop) -> None:
    """Send the loop's unhandled errors through asyncio_handler."""
    loop.set_exception_handler(asyncio_handler)


def redacting_sink(sink: Callable[[str, str, int, int], None]) -> Callable[[str, str, int, int], None]:
    """Wrap a feed sink so each line's text is masked before it reaches the recorder or the engines."""
    def wrapped(kind: str, raw: str, recv_ms: int, conn_id: int) -> None:
        sink(kind, redact(raw), recv_ms, conn_id)
    return wrapped
