# Prepared algorithms

```cpp
auto exec = iane::make_execution_context(policy);
auto forward = exec.prepare(iane::fft_forward, spatial.const_view(), spectrum.view());
auto inverse = exec.prepare(iane::fft_inverse, spectrum.const_view(), spatial.view());
auto haar = exec.prepare(iane::wavelet_forward, samples.const_view(), bands.view());

exec.run(forward, spatial.const_view(), spectrum.view());
exec.run(inverse, spectrum.const_view(), spatial.view());
exec.run(haar, samples.const_view(), bands.view()).wait();
```

Include `<iane/algorithms/fft/fft.hpp>` and `<iane/algorithms/wavelet/haar.hpp>`.
The context reserves shared temporary storage during preparation. Prepared
operations belong to that context. They can run with other views of the same
shape and supported layout. CPU execution completes before returning. A CUDA
context borrows one stream and uses its ordering to reuse scratch between
successive submissions. Keep the views and prepared operations alive until
completion. Destroy prepared operations before their context, and destroy a
CUDA context before its stream. Submit to a context from one host thread at
a time.
Use separate contexts for overlapping work. Growing the workspace on CUDA
waits for previous work before replacing the allocation.

FFT is single-precision, out-of-place 2D complex-to-complex over packed memory
views of `iane::fft_complex`. Forward uses the negative exponent. Inverse is
unnormalized, so forward followed by inverse multiplies the input by the
number of pixels. FFTW uses a single-threaded estimate plan and accepts new
buffers without imposing FFTW-specific SIMD alignment. CUDA uses cuFFT with
caller-managed working memory supplied by the context. Enable FFT with
`-DIANE_ENABLE_FFT=ON` and link the `iane_fft` CMake target.

The single-level Haar transform accepts readable float inputs and writable
float outputs with matching positive even 2D extents. It applies orthonormal
pair sums and differences horizontally and then vertically. The output
quadrants are low/low, high-horizontal/low-vertical, low-horizontal/high-vertical,
and high/high, in reading order. The horizontal pass fills a temporary float
image before the vertical pass writes output, so in-place execution is allowed.

See [Spectral views and wavelets](spectral_views_and_wavelets.md) for lazy
spectrum display and the db2 and sym4 analysis and synthesis operations.
