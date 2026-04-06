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

        from vllm.config import ProfilerConfig
        # ProfilerConfig field names vary across vLLM versions; build kwargs
        # only from fields that actually exist in this install.
        try:
            valid = set(ProfilerConfig.model_fields.keys())  # pydantic v2
        except AttributeError:
            valid = set(getattr(ProfilerConfig, "__dataclass_fields__", {}).keys())

        candidate = {
            "profiler": "torch",
            "torch_profiler_dir": trace_dir,
            "torch_profiler_record_shapes": prof_cfg.get("trace_record_shapes", True),
            "torch_profiler_with_memory": prof_cfg.get("trace_with_memory", True),
            "torch_profiler_with_stack": prof_cfg.get("trace_with_stack", False),
            "torch_profiler_with_flops": prof_cfg.get("trace_with_flops", True),
            "torch_profiler_use_gzip": prof_cfg.get("trace_use_gzip", True),
        }
        pc_kwargs = {k: v for k, v in candidate.items() if k in valid}
        print(f"ProfilerConfig fields available: {sorted(valid)}")
        print(f"ProfilerConfig kwargs used: {pc_kwargs}")
        llm_kwargs["profiler_config"] = ProfilerConfig(**pc_kwargs)

    sampling_params = SamplingParams(
        temperature=model_cfg.get("temperature", 0.0),
        max_tokens=model_cfg.get("max_new_tokens", 512),
    )

    print(f"Loading {method_name}: {model_path}")
    if profile:
        print(f"Profiling enabled — traces will be saved to {trace_dir}")

    llm = LLM(**llm_kwargs)

    return llm, sampling_params
