"""
Benchmark runner. Loads a model, warms up, runs prompts at each concurrency
level, captures latency/throughput/quality, writes CSVs.

Usage:
    python benchmark/runner.py --method awq_int4 --gpu A100_SXM
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import BenchmarkResult, RequestMetrics
from benchmark.model_loader import load_model
from benchmark.quality_eval import evaluate
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


def _extract_request_stats(output, n_tokens):
    """Extract per-request latency metrics from vLLM RequestOutput.metrics."""
    stats = getattr(output, "metrics", None)

    # Debug: print available metrics fields on first call
    if not hasattr(_extract_request_stats, "_debugged"):
        _extract_request_stats._debugged = True
        print(f"  [DEBUG] metrics is None: {stats is None}")
        print(f"  [DEBUG] output type: {type(output)}")
        print(f"  [DEBUG] output attrs: {[a for a in dir(output) if not a.startswith('_')]}")
        if stats is not None:
            print(f"  [DEBUG] metrics type: {type(stats)}")
            print(f"  [DEBUG] metrics dict: {vars(stats) if hasattr(stats, '__dict__') else 'no __dict__'}")

    if stats is None:
        return float("nan"), float("nan"), 0.0

    # vLLM RequestMetrics fields:
    #   first_token_time, first_scheduled_time, finished_time
    ft = getattr(stats, "first_token_time", 0.0) or 0.0
    st = getattr(stats, "first_scheduled_time", 0.0) or 0.0
    fin = getattr(stats, "finished_time", 0.0) or 0.0

    # TTFT: time from scheduling to first token
    ttft = (ft - st) if ft > 0 and st > 0 else float("nan")

    # End-to-end latency: time from scheduling to last token
    e2e = (fin - st) if fin > 0 and st > 0 else float("nan")

    # Per-request TPS
    tps = n_tokens / e2e if (not math.isnan(e2e) and e2e > 0) else 0.0

    return ttft, e2e, tps


def _run_batch(llm, sampling_params, prompts, formatted_prompts, concurrency,
               request_timeout_sec=None):
    print(f"\n[_run_batch] START concurrency={concurrency} total_prompts={len(prompts)}")
    all_metrics = []
    total_wall = 0.0
    total_tokens = 0

    for start in range(0, len(prompts), concurrency):
        batch_prompts = prompts[start:start + concurrency]
        batch_formatted = formatted_prompts[start:start + concurrency]
        batch_idx = start // concurrency

        t0 = time.perf_counter()
        outputs = llm.generate(batch_formatted, sampling_params)
        batch_time = time.perf_counter() - t0

        batch_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
        total_wall += batch_time
        total_tokens += batch_tokens

        if batch_idx == 0:
            print(f"[_run_batch] First batch: {len(outputs)} outputs, "
                  f"batch_time={batch_time:.3f}s, batch_tokens={batch_tokens}")
            if outputs:
                o = outputs[0]
                print(f"[_run_batch] First output type: {type(o)}")
                print(f"[_run_batch] First output attrs: {[a for a in dir(o) if not a.startswith('_')]}")
                print(f"[_run_batch] First output.outputs[0] type: {type(o.outputs[0])}")
                print(f"[_run_batch] First output.outputs[0] attrs: {[a for a in dir(o.outputs[0]) if not a.startswith('_')]}")

        for i, output in enumerate(outputs):
            prompt = batch_prompts[i]
            text = output.outputs[0].text
            n_tokens = len(output.outputs[0].token_ids)

            ttft, e2e, tps = _extract_request_stats(output, n_tokens)

            # Flag requests that exceeded the configured timeout
            timed_out = (request_timeout_sec and not math.isnan(e2e)
                         and e2e > request_timeout_sec)
            if timed_out:
                print(f"  WARNING: prompt {prompt['id']} exceeded timeout "
                      f"({e2e:.1f}s > {request_timeout_sec}s)")

            eq = evaluate(text, prompt["quality_checks"])

            # LLM-as-Judge scoring
            jr = llm_judge(prompt, text)
            judge_score = jr.score
            judge_passed = jr.passed

            all_metrics.append(RequestMetrics(
                prompt_id=prompt["id"],
                category=prompt["category"],
                output_tokens=n_tokens,
                quality_passed=eq.passed if not timed_out else False,
                quality_pass_rate=eq.pass_rate if not timed_out else 0.0,
                judge_score=judge_score if not timed_out else 0.0,
                judge_passed=judge_passed if not timed_out else False,
                ttft_sec=ttft,
                e2e_latency_sec=e2e,
                tps=tps,
            ))

    return all_metrics, total_wall, total_tokens


# CSV column definitions
DETAIL_COLS = [
    "method", "gpu", "concurrency", "prompt_id", "category",
    "ttft_sec", "e2e_latency_sec",
    "output_tokens", "tps",
    "quality_passed", "quality_pass_rate",
    "judge_score", "judge_passed",
]
SUMMARY_COLS = [
    "method", "gpu", "concurrency", "num_requests",
    "ttft_mean_sec", "ttft_p50_sec", "ttft_p95_sec",
    "e2e_latency_mean_sec", "e2e_latency_p50_sec", "e2e_latency_p95_sec",
    "tps_per_request_mean", "tps_system", "requests_per_sec",
    "quality_pass_rate", "judge_score_mean", "judge_pass_rate",
    "cost_per_1m_tokens_usd",
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
                    "ttft_sec": round(req.ttft_sec, 6) if req.has_ttft else "",
                    "e2e_latency_sec": round(req.e2e_latency_sec, 6) if req.has_e2e else "",
                    "output_tokens": req.output_tokens,
                    "tps": round(req.tps, 2),
                    "quality_passed": req.quality_passed,
                    "quality_pass_rate": round(req.quality_pass_rate, 4),
                    "judge_score": round(req.judge_score, 4),
                    "judge_passed": req.judge_passed,
                })


def _write_summary_csv(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
        w.writeheader()
        for result in results:
            w.writerow(result.summary_row())


def run_benchmark(method_name, gpu, config, profile=False):
    print(f"\n[run_benchmark] START method={method_name} gpu={gpu} profile={profile}")
    bench_cfg = config["benchmark"]
    output_cfg = config["output"]
    concurrency_levels = bench_cfg["concurrency_levels"]
    request_timeout = bench_cfg.get("request_timeout_sec")
    print(f"[run_benchmark] concurrency_levels={concurrency_levels} request_timeout={request_timeout}")

    # Cost config
    cost_cfg = config.get("cost", {})
    hourly_rates = cost_cfg.get("runpod_hourly_rates", {})
    hourly_rate = hourly_rates.get(gpu, 0.0)
    print(f"[run_benchmark] hourly_rate={hourly_rate}")

    prompts_path = PROJECT_ROOT / output_cfg["prompts_file"]
    with open(prompts_path) as f:
        prompts = json.load(f)["prompts"]
    print(f"[run_benchmark] Loaded {len(prompts)} prompts from {prompts_path}")

    llm, sampling_params = load_model(method_name, config, profile=profile)
    print(f"[run_benchmark] Model loaded. sampling_params={sampling_params}")

    tokenizer = llm.get_tokenizer()
    formatted = [_format_prompt(p, tokenizer) for p in prompts]

    _warmup(llm, sampling_params, formatted, bench_cfg.get("warmup_requests", 5))

    all_results = []
    for conc in concurrency_levels:
        print(f"\n--- Concurrency {conc} ---")

        if profile:
            llm.start_profile()

        metrics, wall_time, tokens = _run_batch(
            llm, sampling_params, prompts, formatted, conc,
            request_timeout_sec=request_timeout,
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
            f"TTFT mean={s['ttft_mean_sec']}s p95={s['ttft_p95_sec']}s | "
            f"E2E mean={s['e2e_latency_mean_sec']}s p95={s['e2e_latency_p95_sec']}s | "
            f"TPS system={s['tps_system']} req/s={s['requests_per_sec']} | "
            f"Quality={s['quality_pass_rate']} Judge={s['judge_score_mean']} | "
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
    print("[main] runner.py starting")
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
