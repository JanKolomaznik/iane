#pragma once

#include <iane/concepts/core.hpp>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace iane {

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

/**
 * @ingroup iane_concepts
 * @brief An element exposing byte-valued red, green and blue channels.
 * @tparam TPixel Candidate exposing r, g and b members readable as bytes.
 *
 * The channels represent values in [0, 255]. This contract is independent
 * of layout, so it also applies to pixels with extra channels.
 */
template<class TPixel>
concept Rgb8Element = requires(const TPixel& aPixel) {
	requires std::same_as<std::remove_cvref_t<decltype(aPixel.r)>, std::uint8_t>;
	requires std::same_as<std::remove_cvref_t<decltype(aPixel.g)>, std::uint8_t>;
	requires std::same_as<std::remove_cvref_t<decltype(aPixel.b)>, std::uint8_t>;
};

/**
 * @ingroup iane_concepts
 * @brief A sample and coefficient can be multiplied, accumulated, and converted to the result.
 * @tparam TSample Source sample type. Default initialization supplies the zero border sample.
 * @tparam TCoefficient Kernel coefficient type. The product is sample * coefficient.
 * @tparam TResult Destination element type, explicitly constructible from the sum.
 *
 * Arithmetic pairs accumulate in double precision. For custom pairs, the
 * product type supplies a default-constructed zero and supports assignment
 * from sum + product. For example, a vector pixel may multiply by a scalar
 * weight and add another weighted vector without converting to a scalar.
 */
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

} // namespace iane
