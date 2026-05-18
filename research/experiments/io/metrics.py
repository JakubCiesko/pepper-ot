from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from datetime import UTC
from datetime import datetime
from statistics import mean
from time import perf_counter


@dataclass
class StageMetrics:
    """Counters and timing data for one experiment phase.

    Attributes:
        stage: Name of the workflow phase being measured.
        started_at_utc: ISO timestamp when the metrics object was created.
        ended_at_utc: ISO timestamp set by finish.
        duration_s: Wall-clock duration in seconds set by finish.
        ok: Number of successful items.
        failed: Number of failed items.
        skipped: Number of skipped items.
        errors_by_type: Counts keyed by error or skip reason.
        latencies_s: Per-item latencies for ok and failed items when provided.
    """

    stage: str
    started_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    ended_at_utc: str | None = None
    duration_s: float = 0.0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    errors_by_type: dict[str, int] = field(default_factory=dict)
    latencies_s: list[float] = field(default_factory=list)
    _t0: float = field(default_factory=perf_counter)

    def finish(self) -> None:
        """Mark the stage as complete and record duration/end timestamp."""
        self.duration_s = perf_counter() - self._t0
        self.ended_at_utc = datetime.now(UTC).isoformat(timespec="seconds")

    def record_ok(self, latency_s: float | None = None) -> None:
        """Record one successful item and optional latency."""
        self.ok += 1
        if latency_s is not None:
            self.latencies_s.append(float(latency_s))

    def record_failed(self, error_type: str, latency_s: float | None = None) -> None:
        """Record one failed item, categorized by error type."""
        self.failed += 1
        self.errors_by_type[error_type] = self.errors_by_type.get(error_type, 0) + 1
        if latency_s is not None:
            self.latencies_s.append(float(latency_s))

    def record_skipped(self, reason: str) -> None:
        """Record one skipped item, categorized by reason."""
        self.skipped += 1
        self.errors_by_type[reason] = self.errors_by_type.get(reason, 0) + 1

    def to_dict(self) -> dict:
        """Serialize stage counters, timing, latency, and throughput metrics."""
        total = self.ok + self.failed + self.skipped
        avg_latency = mean(self.latencies_s) if self.latencies_s else 0.0
        throughput = (self.ok / self.duration_s) if self.duration_s > 0 else 0.0
        return {
            "stage": self.stage,
            "started_at_utc": self.started_at_utc,
            "ended_at_utc": self.ended_at_utc,
            "duration_s": self.duration_s,
            "total_items": total,
            "ok": self.ok,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors_by_type": self.errors_by_type,
            "avg_latency_s": avg_latency,
            "throughput_items_per_s": throughput,
        }
