"""Small dependency-free metrics registry for the web process."""

from __future__ import annotations

import threading
import time
from collections import defaultdict


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._requests = defaultdict(int)
        self._latency_sum = defaultdict(float)
        self._latency_count = defaultdict(int)

    def observe_request(self, method: str, path: str, status: int, elapsed_seconds: float) -> None:
        # Prefer the framework route template so IDs do not create unbounded
        # metric labels. Middleware falls back to the raw path for 404s.
        route = path.split("?")[0]
        key = (str(method).upper(), route, str(int(status)))
        with self._lock:
            self._requests[key] += 1
            self._latency_sum[key] += max(0.0, float(elapsed_seconds))
            self._latency_count[key] += 1

    def render_prometheus(self) -> str:
        lines = [
            "# TYPE smart_agent_http_requests_total counter",
            "# TYPE smart_agent_http_request_duration_seconds summary",
        ]
        with self._lock:
            items = tuple(self._requests.items())
            sums = dict(self._latency_sum)
            counts = dict(self._latency_count)
        for (method, path, status), value in sorted(items):
            labels = f'method="{method}",path="{path}",status="{status}"'
            lines.append(f"smart_agent_http_requests_total{{{labels}}} {value}")
            lines.append(f"smart_agent_http_request_duration_seconds_sum{{{labels}}} {sums[(method, path, status)]:.6f}")
            lines.append(f"smart_agent_http_request_duration_seconds_count{{{labels}}} {counts[(method, path, status)]}")
        return "\n".join(lines) + "\n"


metrics_registry = MetricsRegistry()


def now() -> float:
    return time.perf_counter()


__all__ = ["MetricsRegistry", "metrics_registry", "now"]
