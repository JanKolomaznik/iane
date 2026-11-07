#pragma once

#include <iane/algorithms/transform.hpp>

#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <utility>
#include <type_traits>

namespace iane {

/// Boundary sampling for convolution.
struct convolution_options {
	border_mode border = border_mode::clamp;
};

namespace detail {

template<class TSample, class TCoefficient>
using convolution_product = std::remove_cvref_t<decltype(
	std::declval<TSample>() * std::declval<TCoefficient>())>;

template<class TSample, class TCoefficient>
inline constexpr bool arithmetic_convolution =
	std::is_arithmetic_v<TSample> && std::is_arithmetic_v<TCoefficient>;

template<class TSample, class TCoefficient>
using convolution_sum = std::conditional_t<arithmetic_convolution<TSample, TCoefficient>,
	double, convolution_product<TSample, TCoefficient>>;

} // namespace detail

namespace concepts {

/// Samples and coefficients form a product that can be accumulated and converted to the output.
template<class TSample, class TCoefficient, class TResult>
concept ConvolutionElements = std::default_initializable<TSample> &&
	requires(TSample aSample, TCoefficient aCoefficient) { aSample * aCoefficient; } &&
	std::default_initializable<detail::convolution_sum<TSample, TCoefficient>> &&
	requires(TSample aSample, TCoefficient aCoefficient,
		detail::convolution_sum<TSample, TCoefficient> aSum) {
		static_cast<TResult>(aSum);
	} &&
	(detail::arithmetic_convolution<TSample, TCoefficient> ||
	 requires(TSample aSample, TCoefficient aCoefficient,
		 detail::convolution_sum<TSample, TCoefficient> aSum) {
		 aSum = aSum + aSample * aCoefficient;
	 });

} // namespace concepts

namespace detail {

// Compute one output sample from the locator and the flipped kernel.
template<class TKernel, class TResult>
struct convolution_filter {
	TKernel kernel;
	typename TKernel::extents_t size;

	IANE_CUDA_HD_WARNING_DISABLE
	template<concepts::Locator TLocator>
	IANE_DECL_HYBRID TResult operator()(const TLocator& aLocator) const
	{
		using sample_t = typename TLocator::value_type;
		using coefficient_t = typename TKernel::value_type;
		constexpr bool arithmetic = arithmetic_convolution<sample_t, coefficient_t>;
		convolution_sum<sample_t, coefficient_t> sum{};
		typename TKernel::coord_t position{};
		do {
			typename TLocator::offset_t offset{};
			for (int i = 0; i < TKernel::dimension; ++i) {
				math::detail::axis(offset, i) =
					math::detail::axis(size, i) / 2 - math::detail::axis(position, i);
			}
			if constexpr (arithmetic) {
				sum = ::fma(static_cast<double>(aLocator[offset]),
					static_cast<double>(kernel[position]), sum);
			} else {
				sum = sum + aLocator[offset] * kernel[position];
			}
		} while (math::detail::advance<TKernel::dimension>(position, size));

		if constexpr (arithmetic && std::same_as<TResult, std::uint8_t>) {
			if (!(sum > 0.0)) {
				return 0;
			}
			if (sum >= 255.0) {
				return 255;
			}
			const auto integer = static_cast<unsigned int>(sum);
			return static_cast<TResult>(integer + (sum - integer >= 0.5));
		} else if constexpr (arithmetic && std::integral<TResult>) {
			constexpr auto lower = std::numeric_limits<TResult>::lowest();
			constexpr auto upper = std::numeric_limits<TResult>::max();
			if (sum != sum) {
				return {};
			}
			if (sum <= static_cast<double>(lower)) {
				return lower;
			}
			if (sum >= static_cast<double>(upper)) {
				return upper;
			}
			return static_cast<TResult>(sum);
		} else {
			return static_cast<TResult>(sum);
		}
	}
};

} // namespace detail

/**
 * @brief Convolve a source with an odd-sized, centered kernel of matching rank.
 * @param aSource Input view and sampling domain.
 * @param aKernel Coefficients in their original orientation. The sample at p - d
 * is multiplied by the coefficient at offset d.
 * @param aDestination Output of the same extent as source, disjoint from both inputs.
 * @param aOptions Sampling rule outside the source view.
 * @param aExecution Execution context for the views' common backend.
 * @return Completion handle. Keep all view storage valid until completion.
 * @throws std::invalid_argument For invalid kernel or mismatched image dimensions.
 *
 * Arithmetic inputs accumulate in double precision. Custom inputs accumulate
 * in the product type with its default value as zero. Arithmetic results
 * written to integral destinations saturate. Byte outputs additionally round
 * to nearest, ties upward. NaN maps to zero for integral outputs.
 */
template<class TSrc, class TKernel, class TDst>
requires (detail::compatible_transform_views<TSrc, TDst> &&
	detail::compatible_transform_views<TKernel, TDst> &&
	concepts::ConvolutionElements<typename TSrc::value_type, typename TKernel::value_type,
		typename TDst::value_type>)
auto convolve(TSrc aSource, TKernel aKernel, TDst aDestination,
	convolution_options aOptions = {}, view_execution<TDst> aExecution = {})
{
	const auto size = aKernel.dimensions();
	for (int i = 0; i < TKernel::dimension; ++i) {
		const auto extent = math::detail::axis(size, i);
		if (extent <= 0 || extent % 2 == 0) {
			throw std::invalid_argument("Convolution requires positive odd kernel dimensions");
		}
	}
	return transform_locator(aSource, aDestination,
		detail::convolution_filter<TKernel, typename TDst::value_type>{aKernel, size},
		aOptions.border, aExecution);
}

} // namespace iane
