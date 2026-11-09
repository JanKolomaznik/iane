# Color conversion and morphology

## Color conversion

```cpp
auto encoded = iane::map_view(rgb8, iane::rgb8_to_unit);
auto linear = iane::map_view(encoded, iane::srgb_to_linear);
auto luminance = iane::map_view(linear, iane::linear_rgb_to_luminance);
iane::copy(luminance, output, policy).wait();
```

Include `<iane/algorithms/color/conversion.hpp>`. The operations also accept
individual pixels and work on both CPU and CUDA. Color triplets use
`math::vec3f`. The operation name specifies the interpretation of the
channels. `rgb8_to_unit` converts named byte channels to the unit scale.
Transfer functions support finite extended-range RGB values by reflecting
the curve for negative inputs. They perform no clipping or byte conversion.

`rgb_to_hsv` and `hsv_to_rgb` use RGB channels in [0, 1]. Hue is measured
in turns, and saturation and value are in [0, 1]. Achromatic RGB produces
zero hue. The inverse wraps finite hue into [0, 1) before conversion.

`linear_rgb_to_xyz` and `xyz_to_linear_rgb` use the sRGB primaries and D65
white with Y normalized to 1. `xyz_to_lab` and `lab_to_xyz` use the same D65
reference white. Lab uses conventional CIE units with L* in [0, 100] for
nominal luminance and signed a* and b*. RGB reconstructed from an out-of-gamut
color can fall outside [0, 1]. Linear luminance is the XYZ Y component.
The sRGB transfer functions and matrices follow [CSS Color 4](https://www.w3.org/TR/css-color-4/#color-conversion-code).

Include `<iane/algorithms/color/grayscale.hpp>` for the existing weighted
RGB byte to gray8 conversion. Its weights apply to the stored byte values.

## Flat morphology

```cpp
auto footprint = iane::box_footprint(iane::math::vec2i{1, 1});
auto exec = iane::make_execution_context(policy);
auto open = exec.prepare(iane::opening(footprint), input, output);
exec.run(open, input, output).wait();
```

Include `<iane/algorithms/morphology/morphology.hpp>`. Opening applies erosion
then dilation. Use `iane::closing(footprint)` for dilation followed by erosion.
The prepared operations reserve one image of scratch in the execution context
and reuse it across runs. Both passes use the selected border rule. On CUDA
they run on the context's stream. Keep the input and output alive through
completion. The source and destination may be the same view because the first
pass reads the source before the second pass writes the destination.

A footprint specifies a nonnegative radius per axis. The box contains every
offset in that radius. Use `make_flat_footprint(radius, predicate)` for another
shape, such as a cross. The pure predicate takes an offset and must select at
least one. Erosion takes the minimum at offsets `p + d`, and dilation takes
the maximum at offsets `p - d`. The reflection matters for asymmetric
footprints. Nonnegative radii must permit a representable width `2r + 1`.

All four operations derive rank from the view types. Use a scalar radius for
1D, or one radius per axis for higher ranks. Source and destination are scalar
arithmetic views of the same shape, type and backend. Erosion and dilation
require disjoint storage. Strided views and subviews are supported. The default
border clamps to the logical view. Set `morphology_options{.border =
border_mode::zero}` for zero padding. Floating-point NaNs are ignored.
If every selected sample is NaN, erosion returns positive infinity and
dilation returns negative infinity.
