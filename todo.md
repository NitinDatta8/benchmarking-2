1. Previously quantization came from different libraries. Now all of them come from llmcompressor library. More reliable benchmarking results. Condense multiple quantization files into a single quantize.py file. Easier config file, setup file and run_benchmark file.
2. Previously we used the same data for calibration that we used for benchmark tests. Now we have seperate calibration data of 50 samples.
3. Add profiling to see where the time is taken
4. Better evaluation criteria with LLM as a judge.