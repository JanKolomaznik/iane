#pragma once

#include <iane/math/coord_traits.hpp>
#include <iane/math/index.hpp>
#include <utility>

namespace iane::resource::detail {

// Check the packed byte strides and allocation size before allocating an image.
template<class TElement, int tDim>
auto packed_layout(const typename math::dim_traits<tDim>::extents_t& aSize)
{
	const auto count = math::detail::element_count<tDim>(aSize);
	if (count > std::numeric_limits<std::size_t>::max() / sizeof(TElement)) {
		throw std::overflow_error("Packed image byte size overflows");
	}

	typename math::dim_traits<tDim>::extents_t strides{};
	std::size_t stride = sizeof(TElement);
	for (int i = 0; i < tDim; ++i) {
		if (stride > std::numeric_limits<int>::max()) {
			throw std::overflow_error("Packed image stride exceeds the supported range");
		}
		math::detail::axis(strides, i) = static_cast<int>(stride);
		if (i + 1 < tDim) {
			const auto extent = static_cast<std::size_t>(math::detail::axis(aSize, i));
			if (extent != 0 && stride > std::numeric_limits<std::size_t>::max() / extent) {
				throw std::overflow_error("Packed image stride overflows");
			}
			stride *= extent;
		}
	}
	return std::pair{count, strides};
}

} // namespace iane::resource::detail
