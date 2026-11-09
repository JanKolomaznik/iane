#pragma once

#include <iane/algorithms/wavelet/orthogonal.hpp>

namespace iane {
namespace detail {

// Supply the four Daubechies low-pass taps in reconstruction order.
struct db2_filter {
	static constexpr int size = 4;

	IANE_DECL_HYBRID static float low(int aTap)
	{
		constexpr float taps[]{0.482962913f, 0.836516304f, 0.224143868f, -0.129409523f};
		return taps[aTap];
	}
};

} // namespace detail

/// Prepare single-level 2D Daubechies db2 analysis with periodic boundaries.
inline constexpr detail::orthogonal_wavelet_algorithm<detail::db2_filter, false> db2_forward{};
/// Prepare reconstruction of db2_forward coefficients, including in-place use.
inline constexpr detail::orthogonal_wavelet_algorithm<detail::db2_filter, true> db2_inverse{};

} // namespace iane
