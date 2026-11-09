# CUDA texture images

Include `<iane/resource/cuda/texture_image.hpp>` in a CUDA translation unit.

```cpp
iane::texture_image<float> texture(input.dimensions());
iane::cuda_policy policy{.stream = stream};
iane::copy(input.const_view(), texture, policy);
iane::convolve(texture.const_view(), device_kernel.const_view(),
	device_output.view(), {}, policy).wait();
```

The upload and convolution are ordered on the same stream. A lookup at
an integer coordinate returns the uploaded texel at that coordinate
without interpolation or normalization. Coordinates are local to the view.
A texture subview keeps the same texture object and starts its coordinates
at the crop origin. The convolution locator applies its border rule at the
logical view boundary.

The owning image wraps a CUDA array and a texture object.
Use `texture_image<float, 1>(width)` for a 1D texture or
`texture_image<float, 3>(iane::math::vec3i{width, height, depth})` for a volume.
The rank defaults to 2 for both `texture_image` and `texture_image_view`.
A 1D view uses scalar coordinates. A 3D view uses `{x, y, z}`. The read-only
view borrows the texture object. Keep the owner, source of an outstanding
upload, and other borrowed storage alive until work completes. Use the
CUDA device that owns the texture. Uploads accept host or CUDA memory
views of the same rank and extents, with contiguous elements in each row.
Rows and 3D slices may have padding. If the 3D slice spacing cannot be
expressed in whole pitched rows, the upload enqueues one copy per slice.
An empty extent requires no CUDA allocation and the matching upload is a no-op.
Only single-channel 8-, 16- and 32-bit integer elements and 32-bit float
elements are supported.

## Fractional coordinates

Texture views model `iane::concepts::InterpolatedView` and expose
`sample(position)` in CUDA device code. The position is a float in 1D
and a vector of floats in 2D and 3D. The origin is the outer corner of
the first texel. In 2D, texel `{x, y}` occupies
`[x, x + 1] × [y, y + 1]`, so `sample({x + 0.5f, y + 0.5f})` samples
the same center as `view[{x, y}]`. The origin `{0.0f, 0.0f}` is the
outer corner of the view, including for nested crops.

For example, for a 1D texture containing `[10, 20]`, point sampling at
`0.0f`, `0.5f`, and `0.75f` yields `10`. Sampling at `1.0f` or
`1.5f` yields `20`. The texture owner uses CUDA point filtering.
Sampling clamps to the edge texel centers of the logical view, so a crop
behaves as an independent image. An empty view or a nonfinite coordinate
returns a zero-initialized value. The concept leaves boundary behavior and
the filtering rule to the view. Its sampled value type may differ from the
integer-indexed texel type.
