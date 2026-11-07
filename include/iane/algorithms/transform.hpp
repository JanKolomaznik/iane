#pragma once

#include <iane/algorithms/for_each.hpp>

namespace iane {
namespace detail {

// Read the source pixel and store the callback result at the same coordinate.
template<class TSource, class TFunction>
struct value_transform {
	TSource source;
	TFunction function;

	IANE_CUDA_HD_WARNING_DISABLE
	template<class TValue>
	IANE_DECL_HYBRID void operator()(TValue& aDestination, const typename TSource::coord_t& aPosition)
	{
		aDestination = function(source[aPosition]);
	}
};

// Pass the source pixel and its coordinate to the callback before storing the result.
template<class TSource, class TFunction>
struct position_transform {
	TSource source;
	TFunction function;

	IANE_CUDA_HD_WARNING_DISABLE
	template<class TValue>
	IANE_DECL_HYBRID void operator()(TValue& aDestination, const typename TSource::coord_t& aPosition)
	{
		aDestination = function(source[aPosition], aPosition);
	}
};

// Pass a read-only source neighborhood to the callback for each output pixel.
template<class TSource, class TFunction>
struct locator_transform {
	TSource source;
	TFunction function;
	border_mode border;

	IANE_CUDA_HD_WARNING_DISABLE
	template<class TValue>
	IANE_DECL_HYBRID void operator()(TValue& aDestination, const typename TSource::coord_t& aPosition)
	{
		aDestination = function(image_locator<TSource>{source, aPosition, border});
	}
};

template<class TSrc, class TDst>
concept compatible_transform_views =
	TSrc::dimension == 2 && TDst::dimension == 2 &&
	((concepts::HostView<TSrc> && concepts::HostView<TDst>) ||
	 (concepts::DeviceView<TSrc> && concepts::DeviceView<TDst>));

template<class TSrc, class TDst>
auto matching_dimensions(TSrc aSource, TDst aDestination)
{
	const auto size = aDestination.dimensions();
	const auto input = aSource.dimensions();
	if (size[0] < 0 || size[1] < 0 || input[0] != size[0] || input[1] != size[1]) {
		throw std::invalid_argument("Source and destination require matching nonnegative dimensions");
	}
	return size;
}

} // namespace detail

/**
 * @brief Assign function(source_pixel) to each destination pixel.
 * @pre Shifted overlapping views are unsupported. CUDA views must remain
 * valid through completion and the callable must support device execution.
 */
template<class TSrc, class TDst, class TFunction>
requires detail::compatible_transform_views<TSrc, TDst>
auto transform(TSrc aSource, TDst aDestination, TFunction aFunction, view_execution<TDst> aExecution = {})
{
	detail::matching_dimensions(aSource, aDestination);
	return for_each_position(aDestination, detail::value_transform<TSrc, TFunction>{aSource, aFunction}, aExecution);
}

/// Assign function(source_pixel, position) at each coordinate. See transform for preconditions.
template<class TSrc, class TDst, class TFunction>
requires detail::compatible_transform_views<TSrc, TDst>
auto transform_position(TSrc aSource, TDst aDestination, TFunction aFunction, view_execution<TDst> aExecution = {})
{
	detail::matching_dimensions(aSource, aDestination);
	return for_each_position(aDestination, detail::position_transform<TSrc, TFunction>{aSource, aFunction}, aExecution);
}

/**
 * @brief Assign function(locator) to each output pixel using the supplied border rule.
 * @pre Source and destination must not overlap. CUDA views must remain
 * valid through completion and the callable must support device execution.
 */
template<class TSrc, class TDst, class TFunction>
requires detail::compatible_transform_views<TSrc, TDst>
auto transform_locator(TSrc aSource, TDst aDestination, TFunction aFunction,
	border_mode aBorder = border_mode::clamp, view_execution<TDst> aExecution = {})
{
	detail::matching_dimensions(aSource, aDestination);
	return for_each_position(aDestination,
		detail::locator_transform<TSrc, TFunction>{aSource, aFunction, aBorder}, aExecution);
}

} // namespace iane
