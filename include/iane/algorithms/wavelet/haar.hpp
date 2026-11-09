#pragma once

#include <iane/algorithms/wavelet/separable.hpp>

namespace iane {
namespace detail {

// Calculate a low- or high-frequency coefficient along one axis.
template<int tAxis, class TSource>
struct haar_pass {
	TSource source;
	int half;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID float operator()(math::vec2i aPosition) const
	{
		const bool high = aPosition[tAxis] >= half;
		aPosition[tAxis] = 2 * (aPosition[tAxis] - (high ? half : 0));
		const float first = source[aPosition];
		++aPosition[tAxis];
		const float second = source[aPosition];
		return (high ? first - second : first + second) * 0.7071067811865475f;
	}
};

// Recombine the low and high coefficients for one output pixel.
template<int tAxis, class TSource>
struct haar_inverse_pass {
	TSource source;
	int half;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID float operator()(math::vec2i aPosition) const
	{
		const bool odd = aPosition[tAxis] % 2 != 0;
		aPosition[tAxis] /= 2;
		const float low = source[aPosition];
		aPosition[tAxis] += half;
		const float high = source[aPosition];
		return (odd ? low - high : low + high) * 0.7071067811865475f;
	}
};

} // namespace detail

/// Prepare a single-level orthonormal 2D Haar transform, including in-place use.
inline constexpr detail::separable_wavelet_algorithm<detail::haar_pass> wavelet_forward{};

/// Prepare single-level synthesis of Haar coefficients, including in-place use.
inline constexpr detail::separable_wavelet_algorithm<detail::haar_inverse_pass, false> wavelet_inverse{};

} // namespace iane
