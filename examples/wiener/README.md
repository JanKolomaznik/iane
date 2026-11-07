# Wiener deconvolution

Build with `-DIANE_BUILD_EXAMPLES=ON -DIANE_ENABLE_FFT=ON`. The example uses
FFTW on CPU and cuFFT on CUDA. Set `-DIANE_ENABLE_CUDA=OFF` for a CPU-only build.

```sh
./build/examples/wiener blurred.png restored.png --psf psf.txt --noise 0.002 --backend cpu
```

The input is decoded as grayscale. The output is a grayscale image of the same
size. The PSF file starts with its positive odd width and height followed by
nonnegative finite coefficients in row-major order. Its sum must be positive
and the PSF must fit in the image. For example, a horizontal blur is:

```text
3 1
1 2 1
```

The PSF is normalized to sum to one and shifted so its center is at the FFT
origin. The method assumes circular boundaries and can exhibit wraparound at
image edges. Each frequency is multiplied by the conjugate PSF response divided
by the response power plus the positive regularization constant `--noise`.
The inverse is normalized and its real part is rounded and clipped to 8 bits.

Use `--help` for all options. `--config` accepts a JSON object with long option
names, and explicit CLI options override the file.

On CUDA, `setup_ms`, `upload_ms`, and `download_ms` cover staging allocations
and transfers. `prepare_ms` includes work-image allocation, FFT planning,
scratch reservation and the PSF FFT. `compute_ms` reports min, median and max
wall time through completion for the image FFT, filtering, inverse FFT and
conversion. Warm-up runs are excluded. File I/O and host PSF preparation
are outside these timings.

The spectral filter follows the [BoltView Wiener example](https://github.com/eyense/boltview/blob/dev/examples/wiener/doc/wiener.md).
