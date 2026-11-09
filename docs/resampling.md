# Resampling and sampled neighborhoods

```cpp
auto source = iane::make_interpolated_view(
    input.const_view(), iane::linear, iane::border_mode::clamp);
iane::resample(source, output.view(), output_to_input, policy).wait();
```

Include `<iane/algorithms/resample.hpp>` and
`<iane/view/interpolated_view.hpp>`. The map receives the center of each
destination pixel in continuous destination coordinates and returns a
continuous position in the source. Coordinates start at the outer corner of
each view. The center of index `i` is `i + 0.5` along each axis. An identity
map preserves pixel centers. For example, resizing a 1D source of width 4 to
width 2 with the map `p -> 2 * p` samples positions 1 and 3. For source values
`[10, 20, 30, 40]`, linear interpolation yields `[15, 35]`.

The views share a backend, and the source and destination may have different
ranks. For example, a map from a 2D point to a 3D point can extract a plane from
a volume. The map must be callable on the selected backend and return a
position compatible with the source's `sample_coord_t`. Destination
coordinates use the float coordinate type for its rank. Resampling visits the
destination once, uses no intermediate image and follows the source sampler's
interpolation and boundary rules. A CUDA texture view can be passed directly.
Keep borrowed storage alive through completion and use disjoint source and
destination storage with distinct output locations.

The software adapter borrows a readable view and preserves integer indexing.
It also works with strided views, crops and readable expressions. `nearest`
selects `floor(p)` on each axis and returns the source value type. At a shared
edge it selects the higher index. `linear` interpolates scalar values using
neighbors around `p - 0.5` on every axis. It promotes integral input to float
and preserves floating-point input types. Clamp extends the edge texel, while
zero padding supplies zero for each tap outside the view. For example, a
linear sample at the outer corner of a 2D view with zero padding returns one
quarter of the first texel. Empty views and nonfinite sample positions return
zero. The adapter requires nonnegative extents. The caller chooses an output
element type that can represent the result.

```cpp
iane::sampling_image_locator locator(source, center);
auto value = locator.sample(offset);
```

Include `<iane/locator/sampling_image_locator.hpp>`. The locator implements
`iane::concepts::SamplingLocator`. `position()` returns the continuous
center in source coordinates, and `sample(offset)` samples at
`position() + offset`. A zero offset samples the center. The locator borrows
the source and inherits its filtering and boundary rules. It can be used
inside CPU or CUDA callbacks for tasks such as extracting subpixel patches.
