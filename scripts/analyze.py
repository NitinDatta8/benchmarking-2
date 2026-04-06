"""Generate comparison charts from benchmark results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

RESULTS_DIR = Path(__file__).parent.parent / "results"
CHARTS_DIR = RESULTS_DIR / "charts"
PALETTE = sns.color_palette("tab10")

METHOD_ORDER = ["baseline_fp16", "fp8_dynamic", "gptq_w4a16", "awq_w4a16"]
METHOD_LABELS = {
    "baseline_fp16": "FP16 (Baseline)",
    "fp8_dynamic": "FP8 Dynamic",
    "gptq_w4a16": "GPTQ W4A16",
    "awq_w4a16": "AWQ W4A16",
}
METHOD_STYLES = {
    "baseline_fp16": {"linestyle": "-",  "marker": "o"},
    "fp8_dynamic":   {"linestyle": "-",  "marker": "s"},
    "gptq_w4a16":    {"linestyle": "--", "marker": "^"},
    "awq_w4a16":     {"linestyle": "-.", "marker": "D"},
}


def load_summaries():
    files = sorted(RESULTS_DIR.glob("*_summary.csv"))
    if not files:
        raise SystemExit("No summary CSVs found in results/")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["method", "concurrency"])


def label(method):
    return METHOD_LABELS.get(method, method)


def save(fig, name):
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved charts/{name}")


# -- Charts --------------------------------------------------------------------

def chart_latency(df):
    """E2E latency vs concurrency (line chart)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(METHOD_ORDER):
        d = df[df["method"] == m].sort_values("concurrency")
        ax.plot(d["concurrency"], d["e2e_latency_mean_sec"],
                label=label(m), color=PALETTE[i], linewidth=2, markersize=8,
                **METHOD_STYLES[m])
    ax.set(xlabel="Concurrency", ylabel="Mean E2E Latency (sec)",
           title="End-to-End Latency vs Concurrency")
    ax.set_xticks(sorted(df["concurrency"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "latency_vs_concurrency.png")


def chart_throughput(df):
    """System TPS vs concurrency (line chart)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(METHOD_ORDER):
        d = df[df["method"] == m].sort_values("concurrency")
        ax.plot(d["concurrency"], d["tps_system"],
                label=label(m), color=PALETTE[i], linewidth=2, markersize=8,
                **METHOD_STYLES[m])
    ax.set(xlabel="Concurrency", ylabel="System Throughput (tok/sec)",
           title="Throughput vs Concurrency")
    ax.set_xticks(sorted(df["concurrency"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "throughput_vs_concurrency.png")


def chart_cost(df):
    """Cost per 1M tokens vs concurrency (line chart)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(METHOD_ORDER):
        d = df[df["method"] == m].sort_values("concurrency")
        ax.plot(d["concurrency"], d["cost_per_1m_tokens_usd"],
                label=label(m), color=PALETTE[i], linewidth=2, markersize=8,
                **METHOD_STYLES[m])
    ax.set(xlabel="Concurrency", ylabel="Cost per 1M Tokens (USD)",
           title="Inference Cost vs Concurrency")
    ax.set_xticks(sorted(df["concurrency"].unique()))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend()
    ax.grid(True, alpha=0.3)
    save(fig, "cost_vs_concurrency.png")


def chart_quality(df):
    """Quality score bar chart at concurrency=1."""
    d = df[df["concurrency"] == 1].sort_values("judge_score_mean", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [label(m) for m in d["method"]]
    bars = ax.bar(labels, d["judge_score_mean"], color=PALETTE[:len(d)],
                  edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set(ylabel="Mean Judge Score", title="Output Quality by Method (Concurrency=1)")
    ax.set_ylim(0.75, 0.95)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    save(fig, "quality_comparison.png")


def chart_summary_table(df):
    """Grouped bar chart: latency, throughput, cost, quality side-by-side at c=1 and c=16."""
    c1 = df[df["concurrency"] == 1].set_index("method")
    c16 = df[df["concurrency"] == 16].set_index("method")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("Quantization Benchmark Summary — L4 GPU", fontsize=14, fontweight="bold")

    methods = METHOD_ORDER
    x = range(len(methods))
    colors = PALETTE[:len(methods)]

    # Latency at c=1 vs c=16
    ax = axes[0, 0]
    w = 0.35
    ax.bar([i - w/2 for i in x], [c1.loc[m, "e2e_latency_mean_sec"] for m in methods], w,
           label="c=1", color=colors, alpha=0.8, edgecolor="white")
    ax.bar([i + w/2 for i in x], [c16.loc[m, "e2e_latency_mean_sec"] for m in methods], w,
           label="c=16", color=colors, alpha=0.4, edgecolor="white")
    ax.set(ylabel="Latency (sec)", title="E2E Latency")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(m) for m in methods], fontsize=8, rotation=15, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Throughput at c=1 vs c=16
    ax = axes[0, 1]
    ax.bar([i - w/2 for i in x], [c1.loc[m, "tps_system"] for m in methods], w,
           label="c=1", color=colors, alpha=0.8, edgecolor="white")
    ax.bar([i + w/2 for i in x], [c16.loc[m, "tps_system"] for m in methods], w,
           label="c=16", color=colors, alpha=0.4, edgecolor="white")
    ax.set(ylabel="Tokens/sec", title="System Throughput")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(m) for m in methods], fontsize=8, rotation=15, ha="right")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Cost at c=1 vs c=16
    ax = axes[1, 0]
    ax.bar([i - w/2 for i in x], [c1.loc[m, "cost_per_1m_tokens_usd"] for m in methods], w,
           label="c=1", color=colors, alpha=0.8, edgecolor="white")
    ax.bar([i + w/2 for i in x], [c16.loc[m, "cost_per_1m_tokens_usd"] for m in methods], w,
           label="c=16", color=colors, alpha=0.4, edgecolor="white")
    ax.set(ylabel="$/1M Tokens", title="Inference Cost")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(m) for m in methods], fontsize=8, rotation=15, ha="right")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2f"))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Quality (consistent across concurrency)
    ax = axes[1, 1]
    scores = [c1.loc[m, "judge_score_mean"] for m in methods]
    bars = ax.bar(list(x), scores, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set(ylabel="Judge Score", title="Output Quality")
    ax.set_xticks(list(x))
    ax.set_xticklabels([label(m) for m in methods], fontsize=8, rotation=15, ha="right")
    ax.set_ylim(0.75, 0.95)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save(fig, "summary_dashboard.png")


def main():
    df = load_summaries()
    print(f"Loaded {len(df)} rows, generating charts to {CHARTS_DIR} ...\n")

    chart_latency(df)
    chart_throughput(df)
    chart_cost(df)
    chart_quality(df)
    chart_summary_table(df)

    print("\nDone. 5 charts generated.")


if __name__ == "__main__":
    main()
