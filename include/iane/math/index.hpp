#pragma once

#include <iane/config.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace iane::math::detail {

// Treat the scalar representation of a 1D coordinate like a single axis.
template<class TCoordinate>
IANE_DECL_HYBRID decltype(auto) axis(TCoordinate& aCoordinate, int aAxis)
{
	if constexpr (std::is_integral_v<TCoordinate>) {
		return (aCoordinate);
	} else {
		return aCoordinate[aAxis];
	}
}

// Validate the iteration domain and compute its size without wrapping.
template<int tDim, class TExtent>
std::size_t element_count(const TExtent& aSize)
{
	bool empty = false;
	for (int i = 0; i < tDim; ++i) {
		const auto extent = axis(aSize, i);
		if constexpr (std::is_signed_v<decltype(extent)>) {
			if (extent < 0) {
				throw std::invalid_argument("Image dimensions must be nonnegative");
			}
		}
		empty |= extent == 0;
	}
	if (empty) {
		return 0;
	}

	std::size_t count = 1;
	for (int i = 0; i < tDim; ++i) {
		const auto extent = static_cast<std::size_t>(axis(aSize, i));
		if (count > std::numeric_limits<std::size_t>::max() / extent) {
			throw std::overflow_error("Image element count overflows");
		}
		count *= extent;
	}
	return count;
}

// Advance in storage order, with the first coordinate changing fastest.
template<int tDim, class TCoordinate, class TExtent>
IANE_DECL_HYBRID bool advance(TCoordinate& aPosition, const TExtent& aSize)
{
	for (int i = 0; i < tDim; ++i) {
		auto& coordinate = axis(aPosition, i);
		if (++coordinate < axis(aSize, i)) {
			return true;
		}
		coordinate = 0;
	}
	return false;
}

// Decode the linear work index of a CUDA thread.
template<int tDim, class TCoordinate, class TExtent>
IANE_DECL_HYBRID TCoordinate coordinate_at(std::size_t aIndex, const TExtent& aSize)
{
	TCoordinate position{};
	for (int i = 0; i < tDim - 1; ++i) {
		const auto extent = static_cast<std::size_t>(axis(aSize, i));
		axis(position, i) = static_cast<int>(aIndex % extent);
		aIndex /= extent;
	}
	axis(position, tDim - 1) = aIndex;
	return position;
}

} // namespace iane::math::detail
