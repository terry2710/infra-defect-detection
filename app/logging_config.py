"""Structured (one-JSON-object-per-line) logging to stdout.

Plain stdout JSON lines rather than a logging library's default text format, because the
consumer here is a log aggregator (or `docker logs | jq`), not a human watching a terminal - the
12-factor-app convention of treating logs as an event stream. Every HTTP request gets exactly one
log line, written by the middleware in app/main.py, carrying the fields a real incident needs:
a request id to correlate with client-side reports, latency, status code, and - for /predict -
how many detections came back, so "the model suddenly stopped finding defects" would show up as a
visible pattern in the logs before anyone had to open Grafana.
"""
import json
import logging
import sys
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # `extra={...}` fields passed to logger calls land on the record as plain attributes -
        # merge in anything that isn't one of the standard LogRecord fields.
        standard_fields = logging.LogRecord(
            "", 0, "", 0, "", (), None
        ).__dict__.keys()
        for key, value in record.__dict__.items():
            if key not in standard_fields and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Ultralytics and uvicorn.access both log a lot of per-request noise in their own formats;
    # route them through the same JSON handler instead of silencing them, so "is ultralytics
    # warning about something on every request" stays visible to the same log consumer.
    for noisy_logger in ("uvicorn", "uvicorn.error", "uvicorn.access", "ultralytics"):
        lg = logging.getLogger(noisy_logger)
        lg.handlers = [handler]
        lg.propagate = False

    return logging.getLogger("defect_api")
