# color_to_bw

Convert a color image to an 8-bit grayscale image of the same dimensions.
The example uses Boost.Program_options for the CLI and stb for image I/O.
The stb headers are included in `examples/common`.

## Build

```sh
cmake -S . -B build
cmake --build build --target color_to_bw
```

Examples are enabled by default. Use `-DIANE_BUILD_EXAMPLES=OFF` to
disable them or `-DIANE_ENABLE_CUDA=OFF` to build without a CUDA toolchain.
CMake needs to find Boost.Program_options and Boost.JSON when examples are enabled.

## Command

```sh
./build/examples/color_to_bw input.jpg gray.png --backend cuda \
  --weights 0.2126,0.7152,0.0722 --warmup 2 --runs 10
```

The same options can be loaded with `color_to_bw --config settings.json`.
Use long option names as JSON keys. Explicit CLI values override the file.

```json
{"input": "input.jpg", "output": "gray.png", "weights": "0.2126,0.7152,0.0722"}
```

Use `--backend cpu` to run the same conversion in RAM. The CLI provides
`--help` and accepts `--device` for CUDA device selection. The run count
must be positive and the warm-up count must be nonnegative.

stb detects the input format from its contents and decodes to RGB8,
discarding alpha if present. The output extension selects PNG, JPEG, BMP
or TGA. JPEG uses quality 95 and is lossy. BMP stores the grayscale result
as RGB with identical channel values.

The weights apply to the stored R, G and B channel values in that order.
Round the weighted sum to the nearest integer with ties toward positive
infinity and clamp to [0, 255]. The sum uses double precision with the
same fused multiply-add order on CPU and CUDA. Weights must be finite
and small enough that the weighted sum cannot overflow.

## IANE usage

`iane::run_on` is a public utility. It takes host input and output views
and calls the same generic function with views for the selected backend.
For CPU it uses the host views directly. For CUDA it prepares the execution
context and device images, uploads the input once and downloads the last
result. The host output is ready when it returns.

`iane::grayscale` describes a readable view. `iane::copy` materializes
that view into the output in one pass, using the view types to select
the backend at compile time. The execution context supplies the queue
for that operation. The callback returns the completion object so
`run_on` can wait for each invocation.

## Timing

`run_on` reports host wall-clock times in milliseconds. The header includes
the backend, CUDA device when applicable, image size and run counts.

| Field | Measurement |
| --- | --- |
| `setup_ms` | CUDA execution context setup and device image allocation. |
| `upload_ms` | One transfer of the input to CUDA, through completion. |
| `compute_ms` | Minimum, median and maximum across measured invocations. Each sample runs from callback invocation through completion. |
| `download_ms` | One transfer of the final result to RAM, through completion. |

CPU reports only compute timing. Warm-up invocations also wait for
completion and are excluded from the statistics. Image decoding and writing
happen outside `run_on` and its timing.
