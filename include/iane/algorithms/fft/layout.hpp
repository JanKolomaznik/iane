#pragma once

#include <iane/math/coord_traits.hpp>

#include <stdexcept>

namespace iane {

/**
 * @brief Return the packed spectrum extent for a positive 2D real image.
 * @param aRealExtent Original width and height.
 * @return Extent {width / 2 + 1, height}, with axis zero packed.
 * @throws std::invalid_argument If an extent is nonpositive.
 */
inline math::vec2i rfft_extent(math::vec2i aRealExtent)
{
	if (aRealExtent[0] <= 0 || aRealExtent[1] <= 0) {
		throw std::invalid_argument("Real FFT requires positive dimensions");
	}
	return math::vec2i{aRealExtent[0] / 2 + 1, aRealExtent[1]};
}

} // namespace iane
