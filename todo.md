0. Previously quantization came from different libraries. Now all of them come from llmcompressor library. More reliable benchmarking results. 
1. Condense multiple quantization files into a single quantize.py file. Easier config file, setup file and run_benchmark file.
1. Metrics were not correct previously (memory ones). Now more stable. 
2. Previously we used the same data for calibration that we used for benchmark tests. Now we have seperate calibration data of 50 samples.
3. Add profiling to see where the time is taken