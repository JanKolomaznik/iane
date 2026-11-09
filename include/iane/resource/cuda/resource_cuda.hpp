#pragma once

#include <iane/image/image.hpp>
#include <iane/resource/cuda/managed_image.hpp>
#include <iane/resource/packed_layout.hpp>
#include <iane/resource/cuda/texture_image.hpp>
#include <iane/execution/context.hpp>
#include <iane/view/const_device_image_view.hpp>
#include <iane/view/device_image_view.hpp>

#if defined(__CUDACC__)
#include <type_traits>
#include <memory>

namespace iane::resource {

/// CUDA storage for packed N-dimensional images.
template<typename TElement, int tDim>
struct device_image_resource {
	static_assert(tDim > 0);
	static_assert(std::is_trivially_copyable_v<TElement>, "CUDA image storage requires trivially copyable elements");
	using extents_t = typename math::dim_traits<tDim>::extents_t;
	using view_t = device_image_view<TElement, tDim>;
	using const_view_t = const_device_image_view<TElement, tDim>;

	struct deleter {
		void operator()(TElement* aPointer) const noexcept { cudaFree(aPointer); }
	};
	using owner_t = std::unique_ptr<TElement, deleter>;

	static owner_t allocate(const extents_t& aSize)
	{
		const auto count = detail::packed_layout<TElement, tDim>(aSize).first;
		if (count == 0) {
			return {};
		}
		TElement* data = nullptr;
		iane::detail::check_cuda(cudaMalloc(&data,
			count * sizeof(TElement)), "CUDA image allocation");
		return owner_t{data};
	}

	static TElement* pointer(const owner_t& aOwner) { return aOwner.get(); }

	static extents_t strides(const extents_t& aSize, const owner_t&)
	{
		return detail::packed_layout<TElement, tDim>(aSize).second;
	}
};

} // namespace iane::resource

namespace iane {

/// Owning image allocated in CUDA device memory.
template<typename TElement, int tDim = 2>
using device_image = image<TElement, tDim, resource::device_image_resource<TElement, tDim>>;

} // namespace iane
#endif
