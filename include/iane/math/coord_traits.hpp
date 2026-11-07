#pragma once
/**
 * @file coord_traits.hpp
 * @brief Coordinate and dimension-related aliases using iane::math::vector.
 */

#include <iane/config.hpp>
#include <iane/math/vector.hpp>

namespace iane::math {

/**
 * @brief Dimension traits for image/coordinate-related vectors.
 *
 * Provides standard typedefs for diff, extents, and coordinates.
 */
template<int tN>
struct dim_traits {
	/// Signed integral vector - used as an offset vector.
	using diff_t = vector<int, tN>;

	/// Unsigned integral vector - used for defining sizes.
	using extents_t = vector<int, tN>;

	/// Signed integral coordinate vector.
	using coord_t = vector<int, tN>;

	/// Floating-point coordinate vector.
	using float_coord_t = vector<float, tN>;

	/// Helper to construct extents from up to 4 ints (extra args ignored).
	IANE_DECL_HYBRID static constexpr extents_t
	create_extents(int aV0 = 0, int aV1 = 0, int aV2 = 0, int aV3 = 0)
	{
		int vals[4] = { aV0, aV1, aV2, aV3 };
		extents_t out;
		for (int i = 0; i < tN; ++i) {
			out[i] = vals[i];
		}
		return out;
	}
};

/// 1D specialization: use scalar types directly.
template<>
struct dim_traits<1> {
	using diff_t = int;
	using extents_t = int;
	using coord_t = int;
	using float_coord_t = float;

	IANE_DECL_HYBRID static constexpr extents_t
	create_extents(int aV0 = 0, int aV1 = 0, int aV2 = 0, int aV3 = 0)
	{
		(void)aV1;
		(void)aV2;
		(void)aV3;
		return aV0;
	}
};

//------------------------------------------------------------
// Common aliases
//------------------------------------------------------------

using intervalf_t = vector<float, 2>;
using intervald_t = vector<double, 2>;

using vec2f = vector<float, 2>;
using vec3f = vector<float, 3>;
using vec2i = vector<int, 2>;
using vec3i = vector<int, 3>;

using size2_t = vector<int, 2>;
using size3_t = vector<int, 3>;

using Int2 = vector<int, 2>;
using Int3 = vector<int, 3>;
using Float2 = vector<float, 2>;
using Float3 = vector<float, 3>;

} // namespace iane::math
