"""
Benchmark runner. Loads a model, warms up, runs prompts at each concurrency
level, captures latency/throughput/quality, writes CSVs.

Usage:
    python benchmark/runner.py --method awq_int4 --gpu A100_SXM
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import BenchmarkResult, RequestMetrics
from benchmark.model_loader import load_model
from benchmark.llm_judge import judge as llm_judge


def _build_chat_messages(prompt):
    msgs = []
    if prompt.get("system"):
        msgs.append({"role": "system", "content": prompt["system"]})
    if prompt["type"] == "multi_turn":
        msgs.extend(prompt["messages"])
    else:
        msgs.append({"role": "user", "content": prompt["user"]})
    return msgs


def _format_prompt(prompt, tokenizer):
    msgs = _build_chat_messages(prompt)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    # fallback for tokenizers without chat templates
    parts = [f"<|{m['role']}|>\n{m['content']}" for m in msgs]
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def _warmup(llm, sampling_params, formatted_prompts, n):
    print(f"Warmup: {min(n, len(formatted_prompts))} requests...")
    llm.generate(formatted_prompts[:n], sampling_params)


def _run_batch(llm, sampling_params, prompts, formatted_prompts, concurrency):
    all_metrics = []
    total_wall = 0.0
    total_tokens = 0

    for start in range(0, len(prompts), concurrency):
        batch_prompts = prompts[start:start + concurrency]
        batch_formatted = formatted_prompts[start:start + concurrency]

        t0 = time.perf_counter()
        outputs = llm.generate(batch_formatted, sampling_params)
        batch_time = time.perf_counter() - t0

        batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_wall += batch_time
        total_tokens += batch_tokens

        for i, output in enumerate(outputs):
            prompt = batch_prompts[i]
            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)

            # Wall-clock E2E: all requests in a batch run concurrently
            # via continuous batching, so each request's E2E ≈ batch_time.
            e2e = batch_time
            tps = n_tokens / e2e if e2e > 0 else 0.0

            # LLM-as-Judge scoring
            jr = llm_judge(prompt, text)

            all_metrics.append(RequestMetrics(
                prompt_id=prompt["id"],
                category=prompt["category"],
                output_tokens=n_tokens,
                e2e_latency_sec=e2e,
                tps=tps,
                judge_score=jr.score,
            ))

    return all_metrics, total_wall, total_tokens


# CSV column definitions
DETAIL_COLS = [
    "method", "gpu", "concurrency", "prompt_id", "category",
    "e2e_latency_sec", "output_tokens", "tps", "judge_score",
]
SUMMARY_COLS = [
    "method", "gpu", "concurrency", "num_requests",
    "e2e_latency_mean_sec", "e2e_latency_p50_sec", "e2e_latency_p95_sec",
    "tps_system", "judge_score_mean", "cost_per_1m_tokens_usd",
]


def _write_detail_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAIL_COLS)
        w.writeheader()
        for result in results:
            for req in result.requests:
                w.writerow({
                    "method": result.method, "gpu": result.gpu,
                    "concurrency": result.concurrency,
                    "prompt_id": req.prompt_id, "category": req.category,
                    "e2e_latency_sec": round(req.e2e_latency_sec, 6),
                    "output_tokens": req.output_tokens,
                    "tps": round(req.tps, 2),
                    "judge_score": round(req.judge_score, 4),
                })


def _write_summary_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for result in results:
            w.writerow(result.summary_row())


def run_benchmark(method_name, gpu, config, profile=False):
    bench_cfg = config["benchmark"]
    output_cfg = config["output"]
    concurrency_levels = bench_cfg["concurrency_levels"]

    # When profiling, only run lowest + highest concurrency to save time and
    # avoid huge trace files. Traces vary across batch sizes but the two
    # extremes capture single-request vs fully-batched kernel behavior.
    if profile and len(concurrency_levels) > 2:
        concurrency_levels = [min(concurrency_levels), max(concurrency_levels)]
        print(f"Profiling mode: limiting concurrency to {concurrency_levels}")

    profile_max_prompts = config.get("profiling", {}).get("profile_max_prompts", 3)

    # Cost config
    cost_cfg = config.get("cost", {})
    hourly_rates = cost_cfg.get("runpod_hourly_rates", {})
    hourly_rate = hourly_rates.get(gpu, 0.0)

    prompts_path = PROJECT_ROOT / output_cfg["prompts_file"]
    with open(prompts_path) as f:
        prompts = json.load(f)["prompts"]
    print(f"Loaded {len(prompts)} prompts")

    llm, sampling_params = load_model(method_name, config, profile=profile)

    tokenizer = llm.get_tokenizer()
    formatted = [_format_prompt(p, tokenizer) for p in prompts]

    _warmup(llm, sampling_params, formatted, bench_cfg.get("warmup_requests", 5))

    all_results = []
    for conc in concurrency_levels:
        print(f"\n--- Concurrency {conc} ---")

        if profile:
            run_prompts = prompts[:profile_max_prompts]
            run_formatted = formatted[:profile_max_prompts]
            print(f"Profiling: limiting to {len(run_prompts)} prompts")
            llm.start_profile()
        else:
            run_prompts = prompts
            run_formatted = formatted

        metrics, wall_time, tokens = _run_batch(
            llm, sampling_params, run_prompts, run_formatted, conc,
        )

        if profile:
            llm.stop_profile()

        result = BenchmarkResult.from_requests(
            method=method_name, gpu=gpu, concurrency=conc,
            requests=metrics,
            total_wall_time_sec=wall_time, total_output_tokens=tokens,
            hourly_rate=hourly_rate,
        )

        s = result.summary_row()
        print(
            f"E2E mean={s['e2e_latency_mean_sec']}s p95={s['e2e_latency_p95_sec']}s | "
            f"TPS system={s['tps_system']} | "
            f"Judge={s['judge_score_mean']} | "
            f"Cost=${s['cost_per_1m_tokens_usd']}/1M"
        )

        all_results.append(result)

    results_dir = PROJECT_ROOT / output_cfg["results_dir"]
    detail_path = results_dir / f"{method_name}_{gpu}_detail.csv"
    summary_path = results_dir / f"{method_name}_{gpu}_summary.csv"
    _write_detail_csv(detail_path, all_results)
    _write_summary_csv(summary_path, all_results)
    print(f"\nWritten: {detail_path}\n         {summary_path}")

    if profile:
        trace_dir = PROJECT_ROOT / output_cfg["results_dir"] / "traces" / method_name
        print(f"\nFlushing profiler traces to {trace_dir} ...")
        time.sleep(10)
        trace_files = sorted(trace_dir.glob("*.json*"))
        if trace_files:
            print(f"Trace files ({len(trace_files)}):")
            for tf in trace_files:
                print(f"  {tf}")
        else:
            print(f"WARNING: No trace files found in {trace_dir}")
        print(f"\nTo view: open chrome://tracing or https://ui.perfetto.dev/ and load the trace file.")

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    parser.add_argument("--profile", action="store_true", default=False,
                        help="Enable vLLM torch profiler (saves traces to results/traces/)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.method not in config["methods"]:
        print(f"Unknown method '{args.method}'. Valid: {', '.join(config['methods'])}")
        sys.exit(1)

    expected_gpus = config["methods"][args.method].get("target_gpus", [])
    if expected_gpus and args.gpu not in expected_gpus:
        print(f"WARNING: method '{args.method}' targets {expected_gpus}, running on '{args.gpu}' instead.")

    run_benchmark(args.method, args.gpu, config, profile=args.profile)


if __name__ == "__main__":
    main()
