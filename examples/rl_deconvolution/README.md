# Richardson–Lucy deconvolution

Built by default with `IANE_BUILD_EXAMPLES=ON`. It supports CPU and CUDA and
has no FFT dependency. Set `-DIANE_ENABLE_CUDA=OFF` for a CPU-only build.

```sh
./build/examples/rl_deconvolution blurred.png restored.png --psf psf.txt \
    --iterations 20 --epsilon 0.000001 --backend cpu
```

The input is decoded as grayscale. The output is a grayscale image of the same
size. The PSF file starts with its positive odd width and height followed by
nonnegative finite coefficients in row-major order. The coefficients must have
a positive sum and are normalized on load. For example:

```text
3 1
1 2 1
```

Each iteration convolves the current estimate with the PSF, divides the observed
image by the predicted image, convolves that ratio with the flipped PSF, and
multiplies the estimate by the resulting correction. The convolutions use zero
outside the image. The correction is normalized by the flipped PSF's response
to an all-ones image to account for support lost at the boundary. The positive
`--epsilon` floors the starting estimate and denominators. The final estimate
is rounded and clipped to 8 bits.

Use `--help` for all options. `--config` accepts a JSON object with long option
names, and explicit CLI options override the file.

On CUDA, `setup_ms`, `upload_ms`, and `download_ms` cover staging allocations
and transfers. `prepare_ms` includes work-image allocation and the edge
sensitivity convolution. `compute_ms` reports min, median and max wall time
through completion, including estimate initialization, all iterations and
conversion. Each run starts from the input again. Warm-up runs are excluded.
File I/O and host PSF preparation are outside these timings.

The iterative update follows the [BoltView Richardson–Lucy example](https://github.com/eyense/boltview/blob/dev/examples/rl_deconvolution/doc/rl_deconvolution.md).
