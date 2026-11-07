#pragma once
#include <iane/config.hpp>
#include <iane/math/vector.hpp>
#include <iane/math/index.hpp>

namespace iane::math {

/**
 * @brief Compute byte offset for strided memory.
 */
template<typename TVector>
requires (!std::is_integral_v<TVector>)
IANE_DECL_HYBRID
inline std::ptrdiff_t offset_in_strided_memory(const TVector &aStrides, const TVector &aCoords)
{
	std::ptrdiff_t acc = 0;
	for (int i = 0; i < TVector::dim; ++i) {
		acc += std::ptrdiff_t(aStrides[i]) * std::ptrdiff_t(aCoords[i]);
	}
	return acc;
}

/// Compute a byte offset for a scalar 1D coordinate.
IANE_DECL_HYBRID inline std::ptrdiff_t offset_in_strided_memory(int aStride, int aCoordinate)
{
	return static_cast<std::ptrdiff_t>(aStride) * aCoordinate;
}

} // namespace iane::math
