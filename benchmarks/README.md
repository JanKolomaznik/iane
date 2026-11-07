# Execution policy benchmark

The benchmark applies a 48-round integer mixing function to every pixel in a
4096 × 2048 image. It runs the same transform sequentially, with two CPU
threads, with four CPU threads, and on CUDA when a device is available.
It checks each result against the serial output.

```cpp
iane::transform(src, dst, mix).wait();
iane::transform(src, dst, mix, iane::cpu_policy{.threads = 4, .grain_size = 4096}).wait();
iane::transform(d_src, d_dst, mix, iane::cuda_policy{.stream = stream}).wait();
```

Each policy produces the same output. CPU execution completes before return.
CUDA queues work on the supplied stream, which must stay alive until work
completes. The CPU policy groups consecutive coordinates in batches of at most
`grain_size` pixels and uses at most `threads` workers, including the calling
thread. It assigns contiguous groups of batches to workers and creates
temporary C++20 threads per call. Parallel callbacks may run in any order
and must synchronize access to shared state. Each worker receives its own
callable copy.

Build and run the standalone executable in Release mode.

```sh
cmake -S . -B build-bench -DCMAKE_BUILD_TYPE=Release -DIANE_BUILD_BENCHMARKS=ON
cmake --build build-bench --target execution_policy_benchmark
./build-bench/benchmarks/execution_policy_benchmark
```

Add `-DIANE_ENABLE_CUDA=OFF` to build the same executable for CPU only.
Each configuration uses one warm-up and three measured runs. The
`compute_ms` statistics include CPU thread creation and joining or CUDA
launch and completion. CUDA setup and transfers are reported separately.
Input initialization and output comparison are outside the timings.

## Convolution benchmark

`convolution_benchmark` applies the same asymmetric 9 × 9 kernel to a
2560 × 1920 image. It runs serial CPU, CPU with four and eight workers,
CUDA with a linear input, and CUDA with a texture input. Each variant
checks every output pixel against the serial result. The CUDA variants use
the same kernel coefficients, border mode, output layout and launch code.
Texture allocation, texture upload and output download are outside the
compute timing. The benchmark reports them separately.

Build with `-DIANE_BUILD_BENCHMARKS=ON` and run
`./build-bench/benchmarks/convolution_benchmark`.
