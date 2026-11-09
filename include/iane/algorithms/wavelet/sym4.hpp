#pragma once

#include <iane/algorithms/wavelet/orthogonal.hpp>

namespace iane {
namespace detail {

// Supply the eight Symlet low-pass taps in reconstruction order.
struct sym4_filter {
	static constexpr int size = 8;

	IANE_DECL_HYBRID static float low(int aTap)
	{
		constexpr float taps[]{0.032223101f, -0.012603967f, -0.099219544f, 0.297857796f,
			0.803738752f, 0.497618668f, -0.029635528f, -0.075765715f};
		return taps[aTap];
	}
};

} // namespace detail

/// Prepare single-level 2D Symlet sym4 analysis with periodic boundaries.
inline constexpr detail::orthogonal_wavelet_algorithm<detail::sym4_filter, false> sym4_forward{};
/// Prepare reconstruction of sym4_forward coefficients, including in-place use.
inline constexpr detail::orthogonal_wavelet_algorithm<detail::sym4_filter, true> sym4_inverse{};

} // namespace iane
