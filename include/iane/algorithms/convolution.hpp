#pragma once

#include <iane/algorithms/transform.hpp>

#include <cmath>
#include <concepts>
#include <cstdint>

namespace iane {

/// Boundary sampling for convolution.
struct convolution_options {
	border_mode border = border_mode::clamp;
};

namespace detail {

// Compute one output sample from the locator and the flipped kernel.
template<class TKernel, class TResult>
struct convolution_filter {
	TKernel kernel;
	int width;
	int height;

	IANE_CUDA_HD_WARNING_DISABLE
	template<concepts::Locator TLocator>
	IANE_DECL_HYBRID TResult operator()(const TLocator& aLocator) const
	{
		double sum = 0.0;
		for (int y = 0; y < height; ++y) {
			for (int x = 0; x < width; ++x) {
				const typename TLocator::offset_t offset{width / 2 - x, height / 2 - y};
				const typename TKernel::coord_t position{x, y};
				sum = ::fma(static_cast<double>(kernel[position]),
					static_cast<double>(aLocator[offset]), sum);
			}
		}
		if constexpr (std::same_as<TResult, std::uint8_t>) {
			if (!(sum > 0.0)) {
				return 0;
			}
			if (sum >= 255.0) {
				return 255;
			}
			const auto integer = static_cast<unsigned int>(sum);
			return static_cast<TResult>(integer + (sum - integer >= 0.5));
		} else {
			return static_cast<TResult>(sum);
		}
	}
};

} // namespace detail

/**
 * @brief Convolve a scalar 2D source with an odd-sized, centered kernel.
 * @param aSource Input view and sampling domain.
 * @param aKernel Coefficients in their original orientation. Each coefficient
 * at offset d multiplies the source sampled at p - d.
 * @param aDestination Output of the same extent as source. It must not overlap either input.
 * @param aOptions Sampling rule outside the source view.
 * @param aExecution Execution context for the views' common backend.
 * @return Completion handle. Keep all view storage valid until completion.
 * @throws std::invalid_argument For invalid kernel or mismatched image dimensions.
 *
 * Accumulates in double precision in row-major kernel order. Floating-point
 * outputs receive the sum. Byte outputs round to nearest, ties upward, and
 * saturate. NaN maps to zero for byte outputs.
 */
template<class TSrc, class TKernel, class TDst>
requires (detail::compatible_transform_views<TSrc, TDst> &&
	detail::compatible_transform_views<TKernel, TDst> &&
	(std::floating_point<typename TDst::value_type> ||
	 std::same_as<typename TDst::value_type, std::uint8_t>))
auto convolve(TSrc aSource, TKernel aKernel, TDst aDestination,
	convolution_options aOptions = {}, view_execution<TDst> aExecution = {})
{
	const auto size = aKernel.dimensions();
	if (size[0] <= 0 || size[1] <= 0 || size[0] % 2 == 0 || size[1] % 2 == 0) {
		throw std::invalid_argument("Convolution requires positive odd kernel dimensions");
	}
	return transform_locator(aSource, aDestination,
		detail::convolution_filter<TKernel, typename TDst::value_type>{aKernel, size[0], size[1]},
		aOptions.border, aExecution);
}

} // namespace iane
