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
