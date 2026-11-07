# convolution

Apply a user-supplied kernel to a grayscale PNG or a rectangular crop.
The example is built by default. Configure with `-DIANE_BUILD_EXAMPLES=OFF`
to disable examples or `-DIANE_ENABLE_CUDA=OFF` for a CPU-only build.

```sh
cmake -S . -B build
cmake --build build --target convolution
```

## Command

```sh
./build/examples/convolution gray.png filtered.png --kernel kernel.txt --roi 16 8 128 96 \
  --border clamp --backend cuda --device 0 --warmup 2 --runs 10
```

The input is a gray8 PNG, such as the output of `color_to_bw`. Omit
`--roi` to process the whole image or use `--backend cpu` to run on the
host. The output has the dimensions of the selected region.

The kernel file starts with its width and height, both positive and odd.
The remaining values are finite coefficients in row-major order. For
example, this asymmetric kernel makes the orientation observable.

```text
3 3
0 0 0
1 0 0
0 0 0
```

With this kernel, an interior output pixel reads the input pixel to its
right. More generally, for a coefficient at kernel offset `d` relative to
the center, multiply it by the source value at `p - d` for output position
`p`. Coefficients are used as supplied. Accumulate in double precision and convert
to gray8 once per output pixel by rounding to nearest with ties toward
positive infinity and saturating to [0, 255].

The crop defines the boundary of the operation. `clamp` samples the nearest
pixel inside the crop. `zero` supplies zero for any sample outside it.

## IANE usage

The example passes a strided source view and a kernel view as two inputs
to `iane::run_on`. The callback receives both views and the output view in
the selected memory space. `iane::convolve` writes directly to the output
and returns a completion handle. CPU and CUDA use the same call shape.
The traversal algorithms and convolution currently support 2D views.

On CUDA, `run_on` uploads both inputs once and downloads the final output
once. The same kernel allocation is reused across warm-up and measured runs.

## Timing

Report `compute_ms` as minimum, median and maximum over the measured runs.
Each sample ends when the convolution completes. Warm-up runs are excluded.
On CUDA, report `setup_ms`, `upload_ms` for both inputs together, and
`download_ms` separately. File I/O and parsing are outside these timings.
