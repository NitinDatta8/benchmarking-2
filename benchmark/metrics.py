"""Metric capture and aggregation for benchmark runs."""

import statistics
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Per-request metrics
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    prompt_id: str
    category: str
    output_tokens: int
    e2e_latency_sec: float
    tps: float
    judge_score: float = 0.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cost_per_million_tokens(hourly_rate: float, tps_system: float) -> float:
    """Compute $/1M tokens from hourly GPU rate and system throughput."""
    if hourly_rate <= 0 or tps_system <= 0:
        return 0.0
    return (hourly_rate / (tps_system * 3600)) * 1_000_000


# ---------------------------------------------------------------------------
# Aggregate benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    method: str
    gpu: str
    concurrency: int
    num_requests: int

    # Latency
    e2e_latency_mean_sec: float
    e2e_latency_p50_sec: float
    e2e_latency_p95_sec: float

    # Throughput
    tps_system: float

    # LLM judge
    judge_score_mean: float

    # Cost
    cost_per_1m_tokens_usd: float

    requests: list[RequestMetrics] = field(default_factory=list)

    @staticmethod
    def from_requests(
        method: str,
        gpu: str,
        concurrency: int,
        requests: list[RequestMetrics],
        total_wall_time_sec: float,
        total_output_tokens: int,
        hourly_rate: float,
    ) -> "BenchmarkResult":
        e2es = np.array([r.e2e_latency_sec for r in requests])
        if len(e2es) == 0:
            e2e_mean, e2e_p50, e2e_p95 = 0.0, 0.0, 0.0
        else:
            e2e_mean = float(e2es.mean())
            e2e_p50 = float(np.percentile(e2es, 50))
            e2e_p95 = float(np.percentile(e2es, 95))

        tps_system = total_output_tokens / total_wall_time_sec if total_wall_time_sec > 0 else 0.0

        judge_scores = [r.judge_score for r in requests]
        judge_score_mean = statistics.mean(judge_scores) if judge_scores else 0.0

        return BenchmarkResult(
            method=method, gpu=gpu, concurrency=concurrency,
            num_requests=len(requests),
            e2e_latency_mean_sec=e2e_mean,
            e2e_latency_p50_sec=e2e_p50,
            e2e_latency_p95_sec=e2e_p95,
            tps_system=tps_system,
            judge_score_mean=judge_score_mean,
            cost_per_1m_tokens_usd=cost_per_million_tokens(hourly_rate, tps_system),
            requests=requests,
        )

    def summary_row(self) -> dict:
        return {
            "method": self.method,
            "gpu": self.gpu,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            "e2e_latency_mean_sec": round(self.e2e_latency_mean_sec, 4),
            "e2e_latency_p50_sec": round(self.e2e_latency_p50_sec, 4),
            "e2e_latency_p95_sec": round(self.e2e_latency_p95_sec, 4),
            "tps_system": round(self.tps_system, 2),
            "judge_score_mean": round(self.judge_score_mean, 4),
            "cost_per_1m_tokens_usd": round(self.cost_per_1m_tokens_usd, 4),
        }
