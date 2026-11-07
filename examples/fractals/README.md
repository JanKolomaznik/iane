# fractals

Render Mandelbrot, Julia or Burning Ship fractals on the CPU or CUDA.
The example is built by default alongside the other IANE utilities.
It evaluates a pixel functor with `iane::for_each_position` and uses
`iane::run_on` to time rendering and retrieve the output.

## Commands

Render a single Mandelbrot image with a named palette.

```sh
./build/examples/fractals mandelbrot.png --width 1200 --height 800 \
  --bounds -2.5 -1.25 1.25 1.25 --palette ocean --iterations 256
```

Render the sample Julia animation as 24 individual images. The first frame
uses `c`, the last uses `c-end`, and the intermediate parameters are
linearly interpolated.

```sh
./build/examples/fractals --config examples/fractals/julia.json
```

The sample writes `julia_0000.png` through `julia_0023.png`. Supply
`--backend cuda` to override the configured backend. Output paths are
relative to the current working directory. Single images use the exact
output path. The filename extension selects PNG, JPEG, BMP or TGA.

## Presets

Run a preset from the repository root with
`./build/examples/fractals --config examples/fractals/<preset>.json`.
Each preset renders at least 1600 × 1200 pixels and uses one measured
render per frame. Set `--width 400 --height 300` for a quick preview.

| Preset | Resolution | Subject |
| --- | --- | --- |
| `mandelbrot-seahorse` | 1920 × 1440 | A wider crop near the join of the main cardioid and the period-two bulb. |
| `mandelbrot-spiral` | 1920 × 1440 | A tight crop with a higher iteration limit and a violet-to-amber palette. |
| `burning-ship-harbor` | 1600 × 1200 | A crop along the left-hand edge with an ember-colored palette. |
| `julia-branches` | 1600 × 1200 | A still Julia image at `c = -0.4 + 0.6i`. |
| `julia` | 1600 × 1200 | A 24-frame Julia sequence that interpolates the complex parameter. |

## Options

| Option | Meaning |
| --- | --- |
| `type` | `mandelbrot`, `julia` or `burning-ship`. |
| `width`, `height` | Output size in pixels. |
| `bounds` | Complex-plane viewport as `xmin ymin xmax ymax`. Sampling uses pixel centers, with the largest imaginary coordinate at the top. |
| `iterations` | Maximum iteration count per pixel. |
| `palette` | Built-in `ocean`, `ember` or `gray` palette. |
| `colors` | Optional custom palette of 2 to 32 RGB hex colors. It takes precedence over `palette`. |
| `color-period` | Number of escape iterations per palette cycle. Colors interpolate between adjacent stops. |
| `inside` | RGB hex color for pixels that do not escape by the iteration limit. |
| `c`, `c-end` | Julia parameters as real and imaginary components. `c-end` is required for a sequence. |
| `frames` | Defaults to one. A value greater than one produces a Julia sequence. |
| `backend`, `device` | CPU or CUDA backend and optional CUDA device index. |
| `warmup`, `runs` | Number of warm-up and measured runs per frame. |

Configuration files are JSON objects with long option names as keys.
Multi-value options such as `bounds` and `colors` use arrays. Explicit CLI
options override the file, and defaults fill in omitted options. Run
`fractals --help` for the complete CLI and default values.

Each frame logs its path and the minimum, median and maximum render times.
CUDA also reports setup and download. File encoding and writing are
outside the timings.
