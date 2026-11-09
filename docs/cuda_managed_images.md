# CUDA managed images

Include `<iane/resource/cuda/managed_image.hpp>` in a CUDA translation unit.

```cpp
iane::managed_image<float, 2> image(iane::math::vec2i{640, 480});
iane::for_each(image.host_view(), initialize);
iane::for_each(image.device_view(), update, iane::cuda_policy{.stream = stream}).wait();
auto result = std::as_const(image).host_view();
```

`initialize` writes pixels on the host. `update` is a device-callable
operation that updates each pixel. After completion, `result` reads the same
allocation on the host. The two accessors select the execution backend and
return writable views on mutable owners and read-only views on const owners.
Each view supports normal packed indexing and strided subviews.

The move-only owner allocates packed storage with `cudaMallocManaged` and
releases it with `cudaFree`. Empty extents allocate no CUDA memory. The
accessors borrow storage and perform no transfer, prefetch, or synchronization.
Keep the allocation alive and unchanged in size through all queued use.
Synchronize before handing off access between CPU and GPU. On platforms with
limited Unified Memory support, host access may require the GPU to be idle.
CUDA controls migration and residency according to the device and platform.
