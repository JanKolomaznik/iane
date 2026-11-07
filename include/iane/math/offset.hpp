#pragma once
#include <iane/config.hpp>
#include <iane/math/vector.hpp>

namespace iane::math {

/**
 * @brief Compute byte offset for strided memory.
 */
template<typename TVector>
IANE_DECL_HYBRID
inline std::ptrdiff_t offset_in_strided_memory(const TVector &aStrides, const TVector &aCoords)
{
	std::ptrdiff_t acc = 0;
	for (int i = 0; i < TVector::dim; ++i) {
		acc += std::ptrdiff_t(aStrides[i]) * std::ptrdiff_t(aCoords[i]);
	}
	return acc;
}

} // namespace iane::memory
