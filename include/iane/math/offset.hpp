#pragma once
#include <iane/config.hpp>
#include <iane/math/vector.hpp>

namespace iane::math {

/**
 * @brief Compute byte offset for strided memory.
 */
template<typename Vec>
IANE_DECL_HYBRID
inline std::ptrdiff_t offset_in_strided_memory(const Vec &strides, const Vec &coords)
{
	std::ptrdiff_t acc = 0;
	for (int i = 0; i < Vec::dim; ++i) {
		acc += std::ptrdiff_t(strides[i]) * std::ptrdiff_t(coords[i]);
	}
	return acc;
}

} // namespace iane::memory
