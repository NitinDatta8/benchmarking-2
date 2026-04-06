"""Unified quantization script using llm-compressor for all methods.

Usage:
    python scripts/quantize.py --method gptq_w4a16 --base_model /workspace/models/base --output_path /workspace/models/gptq_w4a16
    python scripts/quantize.py --method fp8_dynamic  # no calibration needed
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import yaml


METHODS_REQUIRING_CALIBRATION = {"gptq_w4a16", "awq_w4a16", "nvfp4"}

SCHEME_MAP = {
    "gptq_w4a16": "W4A16",
    "awq_w4a16": "W4A16_ASYM",
    "fp8_dynamic": "FP8_DYNAMIC",
    "nvfp4": "NVFP4",
}


def _build_chat_messages(prompt):
    """Build chat message list from a prompt dict."""
    msgs = []
    if prompt.get("system"):
        msgs.append({"role": "system", "content": prompt["system"]})
    if prompt["type"] == "multi_turn":
        msgs.extend(prompt["messages"])
    else:
        msgs.append({"role": "user", "content": prompt["user"]})
    return msgs


def load_calibration_data(config, tokenizer, num_samples=128):
    """Load calibration texts from dedicated calibration prompts (separate from benchmarks).

    Applies the tokenizer's chat template so calibration activations match
    the format the model will see during real inference.
    """
    calib_file = config["output"].get("calibration_file", "prompts/calibration_prompts.json")
    with open(calib_file) as f:
        data = json.load(f)

    texts = []
    for p in data["prompts"]:
        msgs = _build_chat_messages(p)
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        else:
            parts = [f"<|{m['role']}|>\n{m['content']}" for m in msgs]
            parts.append("<|assistant|>\n")
            text = "\n".join(parts)
        texts.append(text)

    rng = random.Random(42)
    rng.shuffle(texts)
    if len(texts) < num_samples:
        # Sample with replacement to reach num_samples without heavy duplication
        extras = rng.choices(texts, k=num_samples - len(texts))
        texts.extend(extras)
    return texts[:num_samples]


def build_recipe(method):
    """Build the llm-compressor recipe for a given method."""
    scheme = SCHEME_MAP[method]

    if method == "gptq_w4a16":
        from llmcompressor.modifiers.quantization import GPTQModifier
        return GPTQModifier(
            targets="Linear", scheme=scheme, ignore=["lm_head"],
            dampening_frac=0.1,
        )

    if method == "awq_w4a16":
        from llmcompressor.modifiers.awq import AWQModifier
        return AWQModifier(
            targets="Linear", scheme=scheme, ignore=["lm_head"],
            duo_scaling=True,
        )

    from llmcompressor.modifiers.quantization import QuantizationModifier
    return QuantizationModifier(targets="Linear", scheme=scheme, ignore=["lm_head"])


def quantize(method, base_model_path, output_path, config):
    """Quantize a model using the specified method via llm-compressor oneshot()."""
    from llmcompressor import oneshot
    from transformers import AutoModelForCausalLM, AutoTokenizer

    method_cfg = config["methods"][method]
    n_samples = method_cfg.get("calibration_samples", 128)
    seq_len = method_cfg.get("calibration_seqlen", 2048)

    base_model_path = os.path.realpath(base_model_path)
    tag = method.upper()

    print(f"[{tag}] Loading model from {base_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, device_map="auto", torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    recipe = build_recipe(method)

    t0 = time.time()
    if method in METHODS_REQUIRING_CALIBRATION:
        from datasets import Dataset
        calib_texts = load_calibration_data(config, tokenizer, n_samples)
        ds = Dataset.from_dict({"text": calib_texts})
        ds = ds.shuffle(seed=42)
        print(f"[{tag}] Quantizing ({n_samples} calibration samples, seq_len={seq_len})...")
        oneshot(
            model=model,
            recipe=recipe,
            dataset=ds,
            shuffle=True,
            max_seq_length=seq_len,
            num_calibration_samples=n_samples,
        )
    else:
        print(f"[{tag}] Quantizing (no calibration)...")
        oneshot(model=model, recipe=recipe)

    elapsed = time.time() - t0
    print(f"[{tag}] Done in {elapsed:.1f}s")

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path, save_compressed=True)
    tokenizer.save_pretrained(output_path)

    meta = {
        "method": method,
        "base_model": base_model_path,
        "scheme": SCHEME_MAP[method],
        "targets": "Linear",
        "ignored_layers": ["lm_head"],
        "quantization_time_sec": round(elapsed, 1),
    }
    if method in METHODS_REQUIRING_CALIBRATION:
        meta["calibration_samples"] = n_samples
        meta["calibration_seqlen"] = seq_len
    with open(Path(output_path) / "quant_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def push_to_hub(output_path, hf_repo):
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=hf_repo, repo_type="model", exist_ok=True)
    api.upload_folder(folder_path=output_path, repo_id=hf_repo)
    print(f"Pushed to https://huggingface.co/{hf_repo}")


def main():
    parser = argparse.ArgumentParser(description="Unified quantization via llm-compressor")
    parser.add_argument("--method", required=True, choices=list(SCHEME_MAP.keys()),
                        help="Quantization method")
    parser.add_argument("--base_model", default="/workspace/models/base")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--config", default="scripts/benchmark_config.yaml")
    parser.add_argument("--hf_repo", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    quantize(args.method, args.base_model, args.output_path, config)
    if args.hf_repo:
        push_to_hub(args.output_path, args.hf_repo)


if __name__ == "__main__":
    main()
