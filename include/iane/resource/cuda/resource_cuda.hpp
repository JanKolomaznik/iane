#pragma once

#include <iane/image/image.hpp>
#include <iane/execution/context.hpp>
#include <iane/view/const_device_image_view.hpp>
#include <iane/view/device_image_view.hpp>

#if defined(__CUDACC__)
#include <limits>
#include <memory>

namespace iane::resource {

/// CUDA storage for packed two-dimensional images.
template<typename TElement, int tDim>
struct device_image_resource {
	static_assert(tDim == 2, "CUDA image allocation currently supports two dimensions");
	using extents_t = typename math::dim_traits<tDim>::extents_t;
	using view_t = device_image_view<TElement, tDim>;
	using const_view_t = const_device_image_view<TElement, tDim>;

	struct deleter {
		void operator()(TElement* aPointer) const noexcept { cudaFree(aPointer); }
	};
	using owner_t = std::unique_ptr<TElement, deleter>;

	static owner_t allocate(const extents_t& aSize)
	{
		if (aSize[0] < 0 || aSize[1] < 0 ||
		    aSize[0] > std::numeric_limits<int>::max() / static_cast<int>(sizeof(TElement))) {
			throw std::invalid_argument("Invalid CUDA image dimensions");
		}
		if (aSize[0] == 0 || aSize[1] == 0) {
			return {};
		}
		TElement* data = nullptr;
		iane::detail::check_cuda(cudaMalloc(&data,
			static_cast<std::size_t>(aSize[0]) * sizeof(TElement) * aSize[1]), "CUDA image allocation");
		return owner_t{data};
	}

	static TElement* pointer(const owner_t& aOwner) { return aOwner.get(); }

	static extents_t strides(const extents_t& aSize, const owner_t&)
	{
		return extents_t{static_cast<int>(sizeof(TElement)), aSize[0] * static_cast<int>(sizeof(TElement))};
	}
};

} // namespace iane::resource

namespace iane {

/// Owning image allocated in CUDA device memory.
template<typename TElement, int tDim = 2>
using device_image = image<TElement, tDim, resource::device_image_resource<TElement, tDim>>;

} // namespace iane
#endif
