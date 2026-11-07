#pragma once

#include <iane/concepts/view.hpp>
#include <iane/execution/context.hpp>
#include <iane/locator/image_locator.hpp>
#include <iane/math/index.hpp>

#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace iane {
namespace detail {

#if defined(__CUDACC__)
template<class TView, class TFunction>
IANE_CUDA_GLOBAL void visit_positions(TFunction aFunction, std::size_t aCount,
	typename TView::extents_t aSize)
{
	const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
	if (index < aCount) {
		aFunction(math::detail::coordinate_at<TView::dimension, typename TView::coord_t>(index, aSize));
	}
}
#endif

// Adapt coordinate traversal to callbacks that take a pixel and its position.
template<class TView, class TFunction>
struct pixel_visitor {
	TView view;
	TFunction function;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void operator()(const typename TView::coord_t& aPosition)
	{
		function(view[aPosition], aPosition);
	}
};

// Adapt a pixel-and-position callback to an operation that only needs the pixel.
template<class TFunction>
struct value_visitor {
	TFunction function;

	IANE_CUDA_HD_WARNING_DISABLE
	template<class TValue, class TPosition>
	IANE_DECL_HYBRID void operator()(TValue&& aValue, const TPosition&)
	{
		function(static_cast<TValue&&>(aValue));
	}
};

// Build a neighborhood locator at each coordinate for the callback.
template<class TView, class TFunction>
struct locator_visitor {
	TView source;
	TFunction function;
	border_mode border;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void operator()(const typename TView::coord_t& aPosition)
	{
		function(image_locator<TView>{source, aPosition, border});
	}
};

/**
 * @brief Visit each host coordinate.
 * @return An already-completed execution handle.
 */
template<concepts::HostView TView, class TFunction>
requires concepts::ViewDomain<TView>
cpu_execution visit_coordinates(TView aView, TFunction aFunction, cpu_execution = {})
{
	const auto size = aView.dimensions();
	if (math::detail::element_count<TView::dimension>(size) != 0) {
		typename TView::coord_t position{};
		do {
			aFunction(position);
		} while (math::detail::advance<TView::dimension>(position, size));
	}
	return {};
}

#if defined(__CUDACC__)
/**
 * @brief Enqueue a visit to each CUDA coordinate.
 * @return The execution handle for waiting on the submitted work.
 * @pre The callable must support device execution and storage must outlive
 * the operation. Invocations may run concurrently in any order.
 */
template<concepts::DeviceView TView, class TFunction>
requires concepts::ViewDomain<TView>
cuda_execution visit_coordinates(TView aView, TFunction aFunction, cuda_execution aExecution = {})
{
	const auto size = aView.dimensions();
	const auto count = math::detail::element_count<TView::dimension>(size);
	if (count != 0) {
		constexpr unsigned int threads = 256;
		const auto blocks = 1 + (count - 1) / threads;
		if (blocks > 2147483647u) {
			throw std::invalid_argument("CUDA traversal exceeds grid size limit");
		}
		visit_positions<TView><<<static_cast<unsigned int>(blocks), threads, 0, aExecution.stream>>>(
			aFunction, count, size);
		detail::check_cuda(cudaGetLastError(), "CUDA traversal launch");
	}
	return aExecution;
}
#endif

} // namespace detail

/// Execution context deduced from a view's backend tag.
template<concepts::AnyView TView>
using view_execution =
#if defined(__CUDACC__)
	std::conditional_t<concepts::DeviceView<TView>, cuda_execution, cpu_execution>;
#else
	cpu_execution;
#endif

/**
 * @brief Visit each pixel with its value or reference and coordinate.
 * @return Execution handle for waiting on completion.
 * @pre For CUDA, the callable must support device execution and storage must
 * outlive the operation. Invocations may run concurrently in any order.
 */
template<concepts::ViewDomain TView, class TFunction>
auto for_each_position(TView aView, TFunction aFunction, view_execution<TView> aExecution = {})
{
	return detail::visit_coordinates(aView, detail::pixel_visitor<TView, TFunction>{aView, aFunction}, aExecution);
}

/// Visit each pixel with its value or reference.
template<concepts::ViewDomain TView, class TFunction>
auto for_each(TView aView, TFunction aFunction, view_execution<TView> aExecution = {})
{
	return for_each_position(aView, detail::value_visitor<TFunction>{aFunction}, aExecution);
}

/// Visit each position with a read-only neighborhood locator.
template<concepts::ReadableView TView, class TFunction>
auto for_each_locator(TView aView, TFunction aFunction, border_mode aBorder = border_mode::clamp,
	view_execution<TView> aExecution = {})
{
	return detail::visit_coordinates(aView, detail::locator_visitor<TView, TFunction>{aView, aFunction, aBorder}, aExecution);
}

} // namespace iane
