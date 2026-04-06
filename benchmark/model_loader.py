"""Load quantized (or baseline) models via vLLM for benchmarking."""

import os
from pathlib import Path

from vllm import LLM, SamplingParams


def load_model(method_name, config, profile=False):
    method = config["methods"][method_name]
    model_cfg = config["model"]

    if method_name == "baseline_fp16":
        model_path = model_cfg["base_local_path"]
    else:
        model_path = method.get("quantized_output_path") or model_cfg["base_local_path"]

    llm_kwargs = {
        "model": model_path,
        "max_model_len": model_cfg["context_length"],
        "trust_remote_code": True,
        "enforce_eager": model_cfg.get("enforce_eager", False),
        "dtype": model_cfg.get("dtype", "float16"),
    }

    # All llm-compressor quantized models use compressed-tensors format,
    # which vLLM auto-detects. No explicit quantization flag needed.

    if profile:
        llm_kwargs["enforce_eager"] = True
        prof_cfg = config.get("profiling", {})
        project_root = Path(__file__).resolve().parent.parent
        trace_dir = str(project_root / config["output"]["results_dir"] / "traces" / method_name)
        Path(trace_dir).mkdir(parents=True, exist_ok=True)

        # vLLM enables the torch profiler when VLLM_TORCH_PROFILER_DIR is set;
        # the LLM then exposes start_profile()/stop_profile() which the runner
        # wraps around generate().
        os.environ["VLLM_TORCH_PROFILER_DIR"] = trace_dir
        if prof_cfg.get("trace_record_shapes", True):
            os.environ["VLLM_TORCH_PROFILER_RECORD_SHAPES"] = "1"
        if prof_cfg.get("trace_with_memory", True):
            os.environ["VLLM_TORCH_PROFILER_WITH_PROFILE_MEMORY"] = "1"
        if prof_cfg.get("trace_with_stack", False):
            os.environ["VLLM_TORCH_PROFILER_WITH_STACK"] = "1"
        if prof_cfg.get("trace_with_flops", True):
            os.environ["VLLM_TORCH_PROFILER_WITH_FLOPS"] = "1"

    sampling_params = SamplingParams(
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_new_tokens", 512),
    )

    print(f"Loading {method_name}: {model_path}")
    if profile:
        print(f"Profiling enabled — traces will be saved to {trace_dir}")

    llm = LLM(**llm_kwargs)

    return llm, sampling_params
