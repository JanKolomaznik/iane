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
integer coordinate `{x, y}` returns the uploaded texel at that coordinate
without interpolation or normalization. Coordinates are local to the view.
A texture subview keeps the same texture object and starts its coordinates
at the crop origin. The convolution locator applies its border rule at the
logical view boundary.

The owning image wraps a 2D CUDA array and a texture object. The read-only
view borrows the texture object. Keep the owner, source of an outstanding
upload, and other borrowed storage alive until work completes. Use the
CUDA device that owns the texture. Uploads accept host or CUDA memory
views with contiguous elements in each row and optional row padding.
Only single-channel 8-, 16- and 32-bit integer elements and 32-bit float
elements are supported.
