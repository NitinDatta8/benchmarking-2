"""Metric capture and aggregation for benchmark runs.

Uses vLLM per-request RequestStateStats for accurate latency/throughput.
"""

import math
import statistics
from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Per-request metrics (populated from vLLM RequestStateStats)
# ---------------------------------------------------------------------------

@dataclass
class RequestMetrics:
    prompt_id: str
    category: str
    output_tokens: int
    quality_passed: bool
    quality_pass_rate: float

    # Latency (seconds)
    ttft_sec: float           # time to first token
    e2e_latency_sec: float    # end-to-end generation latency (per request)

    # Per-request throughput
    tps: float                # output_tokens / e2e_latency_sec

    # LLM judge scores (0-1 normalized)
    judge_score: float = 0.0
    judge_passed: bool = False

    @property
    def has_ttft(self):
        return not math.isnan(self.ttft_sec)

    @property
    def has_e2e(self):
        return not math.isnan(self.e2e_latency_sec)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _percentile_stats(arr: np.ndarray):
    """Return (mean, p50, p95) for a numpy array, or (0, 0, 0) if empty."""
    if len(arr) == 0:
        return 0.0, 0.0, 0.0
    return float(arr.mean()), float(np.percentile(arr, 50)), float(np.percentile(arr, 95))


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

    # Latency aggregates
    ttft_mean_sec: float
    ttft_p50_sec: float
    ttft_p95_sec: float
    e2e_latency_mean_sec: float
    e2e_latency_p50_sec: float
    e2e_latency_p95_sec: float

    # Throughput
    tps_per_request_mean: float
    tps_system: float
    requests_per_sec: float

    # Quality
    quality_pass_rate: float

    # Cost
    cost_per_1m_tokens_usd: float

    # LLM judge
    judge_score_mean: float = 0.0
    judge_pass_rate: float = 0.0

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
        # --- Latency percentiles ---
        ttfts = np.array([r.ttft_sec for r in requests if r.has_ttft])
        e2es = np.array([r.e2e_latency_sec for r in requests if r.has_e2e])

        ttft_mean, ttft_p50, ttft_p95 = _percentile_stats(ttfts)
        e2e_mean, e2e_p50, e2e_p95 = _percentile_stats(e2es)

        # --- Throughput ---
        tps_vals = [r.tps for r in requests if r.tps > 0]
        tps_per_request_mean = statistics.mean(tps_vals) if tps_vals else 0.0
        tps_system = total_output_tokens / total_wall_time_sec if total_wall_time_sec > 0 else 0.0
        requests_per_sec = len(requests) / total_wall_time_sec if total_wall_time_sec > 0 else 0.0

        # --- Quality ---
        quality_passes = [r.quality_passed for r in requests]
        quality_pass_rate = sum(quality_passes) / len(quality_passes) if quality_passes else 0.0

        # --- LLM Judge ---
        judge_scores = [r.judge_score for r in requests]
        judge_score_mean = statistics.mean(judge_scores) if judge_scores else 0.0
        judge_passes = [r.judge_passed for r in requests]
        judge_pass_rate = sum(judge_passes) / len(judge_passes) if judge_passes else 0.0

        return BenchmarkResult(
            method=method, gpu=gpu, concurrency=concurrency,
            num_requests=len(requests),
            # latency
            ttft_mean_sec=ttft_mean, ttft_p50_sec=ttft_p50, ttft_p95_sec=ttft_p95,
            e2e_latency_mean_sec=e2e_mean, e2e_latency_p50_sec=e2e_p50, e2e_latency_p95_sec=e2e_p95,
            # throughput
            tps_per_request_mean=tps_per_request_mean,
            tps_system=tps_system,
            requests_per_sec=requests_per_sec,
            # quality + cost
            quality_pass_rate=quality_pass_rate,
            judge_score_mean=judge_score_mean,
            judge_pass_rate=judge_pass_rate,
            cost_per_1m_tokens_usd=cost_per_million_tokens(hourly_rate, tps_system),
            requests=requests,
        )

    def summary_row(self) -> dict:
        return {
            "method": self.method,
            "gpu": self.gpu,
            "concurrency": self.concurrency,
            "num_requests": self.num_requests,
            # latency
            "ttft_mean_sec": round(self.ttft_mean_sec, 4),
            "ttft_p50_sec": round(self.ttft_p50_sec, 4),
            "ttft_p95_sec": round(self.ttft_p95_sec, 4),
            "e2e_latency_mean_sec": round(self.e2e_latency_mean_sec, 4),
            "e2e_latency_p50_sec": round(self.e2e_latency_p50_sec, 4),
            "e2e_latency_p95_sec": round(self.e2e_latency_p95_sec, 4),
            # throughput
            "tps_per_request_mean": round(self.tps_per_request_mean, 2),
            "tps_system": round(self.tps_system, 2),
            "requests_per_sec": round(self.requests_per_sec, 2),
            # quality + cost
            "quality_pass_rate": round(self.quality_pass_rate, 4),
            "judge_score_mean": round(self.judge_score_mean, 4),
            "judge_pass_rate": round(self.judge_pass_rate, 4),
            "cost_per_1m_tokens_usd": round(self.cost_per_1m_tokens_usd, 4),
        }
