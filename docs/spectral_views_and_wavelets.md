# Spectral views and wavelets

## FFT display

```cpp
auto display = iane::log_magnitude_view(iane::fftshift_view(spectrum));
iane::copy(display, output, policy).wait();
```

Include `<iane/algorithms/fft/spectrum.hpp>`. The lazy shift places DC at
`{width / 2, height / 2}`, using integer division for odd extents. The log
magnitude is `log1p(hypot(real, imag))`, with zero mapping to zero. Output
values are floating point. Display range and conversion to a pixel format
are left to the caller. The view borrows the spectrum and needs no allocation
or FFT library. It can be composed with other views on the same backend.

For spectra of real inputs stored as `{width / 2 + 1, height}`, first use
`iane::hermitian_spectrum_view(packed, full_extent)` in place of `spectrum`.
The adapter reads the stored nonnegative frequencies along axis zero and
uses the complex conjugate at the opposite frequency for missing bins. It
requires the full extent, which distinguishes odd and even source widths.
The existing complex FFT already returns the full spectrum. FFT planning
and execution are declared in `<iane/algorithms/fft/fft.hpp>`.

## Additional wavelets

```cpp
auto exec = iane::make_execution_context(policy);
auto forward = exec.prepare(iane::db2_forward, input, coefficients);
auto inverse = exec.prepare(iane::db2_inverse, coefficients, restored);
exec.run(forward, input, coefficients);
exec.run(inverse, coefficients, restored).wait();
```

Include `<iane/algorithms/wavelet/db2.hpp>` for Daubechies db2, or
`<iane/algorithms/wavelet/sym4.hpp>` for Symlet sym4. Both provide
single-level orthonormal 2D analysis and synthesis over float views with
positive, even extents. Use `<iane/algorithms/wavelet/haar.hpp>` for the
existing `wavelet_forward` Haar operation. See [Prepared algorithms](prepared_algorithms.md)
for scratch ownership and completion.

The new transforms extend each axis periodically. Analysis starts the
low- and high-pass filters at index `2k` for coefficient `k`. The low-pass
taps are in reconstruction-filter order, and the high-pass taps are the
reversed low-pass taps with alternating signs starting positive. Synthesis
is the transpose of analysis, including the periodic wrap. The output
quadrants in reading order are low/low, high/low, low/high, high/high,
where the first label refers to the horizontal axis. Both passes support
strided views. The context holds a full-size scratch image, allowing exact
in-place execution.

The db2 and sym4 tap values can be cross-checked against the
[PyWavelets coefficient tables](https://raw.githubusercontent.com/PyWavelets/pywt/main/pywt/_extensions/c/wavelets_coeffs.template.h).

## Real FFT

```cpp
const auto size = input.dimensions();
iane::host_image<iane::fft_complex, 2> packed(iane::rfft_extent(size));
auto exec = iane::make_execution_context(policy);
auto forward = exec.prepare(iane::rfft, input, packed.view());
auto inverse = exec.prepare(iane::irfft, packed.const_view(), restored);
exec.run(forward, input, packed.view());
exec.run(inverse, packed.const_view(), restored).wait();

auto full = iane::hermitian_spectrum_view(packed.const_view(), size);
auto display = iane::log_magnitude_view(iane::fftshift_view(full));
iane::copy(display, output, policy).wait();
```

Include `<iane/algorithms/fft/rfft.hpp>` and
`<iane/algorithms/fft/spectrum.hpp>`. Enable `IANE_ENABLE_FFT` and link
`iane_fft`. The snippet uses host views and a CPU policy. For CUDA, use
device images and a CUDA policy throughout.

`rfft` converts a packed 2D float image into a packed complex spectrum with
extent `{width / 2 + 1, height}`. The forward transform uses the negative
exponent. `irfft` reads a valid Hermitian half spectrum and uses the real
destination extent to determine the original shape. Both odd and even
extents work. For example, widths 4 and 5 both store three bins per row,
so retain the original extent when constructing the inverse destination
and the full display view. The inverse is unnormalized, so a round trip
multiplies the original image by `width * height`.

Both operations require disjoint buffers and validate layout and extent at
preparation and execution. On CUDA, real and complex buffers both need
complex-element alignment. The inverse preserves the supplied spectrum by
copying it into the context's scratch before invoking the native plan.
This costs one packed-spectrum copy per inverse, in addition to any cuFFT
work area. Keep the views and plans alive through completion.

## Multilevel wavelet decomposition

```cpp
auto exec = iane::make_execution_context(policy);
auto plan = exec.prepare(
    iane::multilevel_wavelet_forward(iane::db2_forward, 3), input, coefficients);
exec.run(plan, input, coefficients).wait();

auto tree = iane::make_wavelet_pyramid_view(coefficients, 3);
auto horizontal_details = tree.details(iane::wavelet_band::hl);
auto finest_detail = horizontal_details.level(0);
auto approximation = tree.approximation();
```

Include `<iane/algorithms/wavelet/multilevel.hpp>`. Pass the Haar
`wavelet_forward`, `db2_forward`, or `sym4_forward` as the single-level
analysis. At each subsequent level, the transform runs in place in the
previous level's upper-left LL quadrant. The original extent is unchanged.
Both axes must be positive and divisible by two at every requested level.
For example, an 8 × 16 image accepts three levels. The first level produces
4 × 8 detail bands, the third produces 1 × 2 detail bands, and the final
approximation is 1 × 2. Exact in-place decomposition is supported and the
context reuses the largest scratch reservation across all levels.

Each detail orientation has its own `pyramid_view`, ordered finest first.
The tree retains the host-side view descriptors and borrows the coefficient
buffer. The coefficient buffer and the tree must outlive the borrowed
`pyramid_view`. Individual band views can outlive the descriptor table
while their coefficient storage remains valid. Only the coarsest LL is
available as an approximation because finer LL quadrants have been
overwritten by their decompositions. The level count passed to the tree
factory must match the decomposition used to produce the coefficients.
