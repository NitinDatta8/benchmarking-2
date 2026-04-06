# RunPod Steps

## 0. SSH into pod

```
ssh into runpod
cd /workspace
```

## 1. Clone repo

```
git clone https://github.com/NitinDatta8/quantization-benchmarking.git
cd quantization-benchmarking
```

## 2. Create `.env` file

Copy the example and fill in your keys:

```bash
cp .env.example .env
# Edit .env with your actual tokens:
#   HF_TOKEN=hf_...
#   OPENAI_API_KEY=sk-...
```

Both `runpod_setup.sh` and `run_benchmark.sh` automatically source `.env` from the repo root.
The Python benchmark runner also reads `.env` via `python-dotenv`.

## 3. Setup environment

Pick the GPU type for your pod:

```bash
# A100
bash scripts/runpod_setup.sh A100_SXM

# L4
bash scripts/runpod_setup.sh L4

# RTX 5090 (Blackwell)
bash scripts/runpod_setup.sh RTX_5090
```

## 4. Run all methods on a GPU

This quantizes (if needed) and benchmarks every method configured for the GPU:

```bash
bash scripts/run_benchmark.sh A100_SXM
bash scripts/run_benchmark.sh L4
bash scripts/run_benchmark.sh RTX_5090
```

## 5. Run a single method

Pass the method name as the 3rd argument:

```bash
# Just baseline
bash scripts/run_benchmark.sh A100_SXM false baseline_fp16

# Just GPTQ
bash scripts/run_benchmark.sh A100_SXM false gptq_w4a16

# Just AWQ
bash scripts/run_benchmark.sh A100_SXM false awq_w4a16

# Just FP8
bash scripts/run_benchmark.sh L4 false fp8_dynamic

# Just NVFP4 (RTX 5090 only)
bash scripts/run_benchmark.sh RTX_5090 false nvfp4
```

## 6. Skip quantization (benchmark only)

If models are already quantized (or downloaded from HF Hub), skip the quant step:

```bash
bash scripts/run_benchmark.sh A100_SXM true gptq_w4a16
```

## 7. Quantize only (no benchmark)

Run the quantize script directly:

```bash
python scripts/quantize.py --method gptq_w4a16 --base_model /workspace/models/base --output_path /workspace/models/gptq_w4a16 --config scripts/benchmark_config.yaml
python scripts/quantize.py --method awq_w4a16 --base_model /workspace/models/base --output_path /workspace/models/awq_w4a16 --config scripts/benchmark_config.yaml
python scripts/quantize.py --method fp8_dynamic --base_model /workspace/models/base --output_path /workspace/models/fp8_dynamic --config scripts/benchmark_config.yaml
python scripts/quantize.py --method nvfp4 --base_model /workspace/models/base --output_path /workspace/models/nvfp4 --config scripts/benchmark_config.yaml
```

Add `--hf_repo <user/repo>` to push the quantized model to HF Hub.

## 8. Benchmark only (direct runner)

Run the benchmark runner directly for a single method:

```bash
python benchmark/runner.py --method gptq_w4a16 --gpu A100_SXM --config scripts/benchmark_config.yaml
```

## 9. Download pre-quantized models from HF Hub

To benchmark on a different GPU without re-quantizing:

```bash
huggingface-cli download <user>/model-GPTQ --local-dir /workspace/models/gptq_w4a16
huggingface-cli download <user>/model-AWQ --local-dir /workspace/models/awq_w4a16
```

Then run with skip-quant:
```bash
bash scripts/run_benchmark.sh L4 true
```

## 10. Profiling (GPU kernel traces)

Enable vLLM's built-in torch profiler to see where time is spent at the CUDA kernel level.
Traces are saved to `results/traces/{method}/` and can be viewed in [Perfetto UI](https://ui.perfetto.dev/).

**Via shell script** (4th argument):
```bash
bash scripts/run_benchmark.sh A100_SXM true baseline_fp16 true
bash scripts/run_benchmark.sh A100_SXM true gptq_w4a16 true
```

**Via runner directly:**
```bash
python benchmark/runner.py --method baseline_fp16 --gpu A100_SXM --profile
python benchmark/runner.py --method gptq_w4a16 --gpu A100_SXM --profile
```

When `--profile` is enabled:
- `enforce_eager=True` is set automatically (disables CUDA graphs so individual kernels are visible in traces)
- Traces include CUDA kernel timing, memory allocations, FLOPs, and tensor shapes
- A 10-second flush period is added at the end for vLLM to write trace files
- Profiling config (what to capture) is in `scripts/benchmark_config.yaml` under `profiling:`

**Viewing traces:**
1. Open https://ui.perfetto.dev/
2. Drag and drop the `.json.gz` file from `results/traces/{method}/`
3. Compare kernel-level behavior across quantization methods

## GPU x Method Matrix

| Method | A100_SXM | L4 | RTX_5090 |
|--------|----------|-----|----------|
| baseline_fp16 | Y | Y | Y |
| gptq_w4a16 | Y | Y | Y |
| awq_w4a16 | Y | Y | Y |
| fp8_dynamic | Y | Y | Y |
| nvfp4 | - | - | Y |

Results are written to `results/`.
