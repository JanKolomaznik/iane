# spectral_transform

Visualize a 2D real FFT or a multilevel Mallat wavelet transform of a grayscale
image. The example builds by default with the other examples. Configure with
`-DIANE_ENABLE_CUDA=OFF` for CPU only. Enable `-DIANE_ENABLE_FFT=ON` to
include FFT mode and its FFTW/cuFFT dependencies. Wavelet modes build with
or without FFT.

```sh
cmake -S . -B build -DIANE_ENABLE_CUDA=OFF -DIANE_ENABLE_FFT=ON
cmake --build build --target spectral_transform
./build/examples/spectral_transform --help
./build/examples/spectral_transform input.jpg amplitude.png --transform fft --phase-output phase.png
./build/examples/spectral_transform input.jpg mallat.png --transform db2 --levels 3
```

The input is decoded to gray8. The output extension selects PNG, JPEG, BMP or
TGA. PNG preserves the display values. The example saves grayscale previews.

- `--transform fft` saves amplitude to `OUTPUT` and phase to
  `--phase-output`. Omit `--phase-output` to append `_phase` to the output
  stem and keep the extension. Both images have the full input dimensions,
  including odd sizes, with DC centered. The amplitude image maps
  `log1p(abs(spectrum))` from zero to the image maximum. Phase maps
  `[-π, π]` to `[0, 255]`. Exactly zero magnitude uses the zero-angle gray.
- `--transform haar` (default), `db2` and `sym4` save one packed Mallat
  coefficient image. `--levels` defaults to 2. The input is extended at
  the right and bottom with edge pixels to the next multiple of
  `2^levels`. The output has these padded dimensions. The upper-left
  approximation is divided by `2^levels` and clamped to byte intensity.
  Each detail band is scaled independently by its maximum absolute
  coefficient. Negative coefficients are darker, positive coefficients
  lighter and zero maps to 128. Bands at the floating-point noise floor
  are filled with 128.

Use `--backend cpu` or `--backend cuda`, with `--device` to select the GPU.
The JSON presets in this folder use the shared CLI. Run from the repository
root and supply your image path. Relative paths are interpreted from the
working directory. Explicit arguments override preset values.

```sh
./build/examples/spectral_transform --config examples/spectral_transform/fft.json --input input.jpg
./build/examples/spectral_transform --config examples/spectral_transform/haar.json --input input.jpg
./build/examples/spectral_transform --config examples/spectral_transform/db2.json --input input.jpg
./build/examples/spectral_transform --config examples/spectral_transform/sym4.json --input input.jpg
```

The shared runner reports `prepare_ms` for plan setup and scratch reservation
and `compute_ms` as minimum, median and maximum transform time through
completion. Warm-up runs are excluded. CUDA additionally reports
`setup_ms`, `upload_ms` and `download_ms`. Image I/O, grayscale-to-float
conversion, padding and display mapping run on the host outside the timed
transform. CUDA transfers the packed FFT spectrum or wavelet coefficients
once after the measured runs.
