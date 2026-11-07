#pragma once

#include <iane/concepts/view.hpp>
#include <iane/execution/context.hpp>
#include <iane/locator/image_locator.hpp>
#include <iane/math/index.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

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
cpu_execution visit_coordinates(TView aView, TFunction aFunction, cpu_policy aPolicy = {})
{
	if (aPolicy.threads == 0 || aPolicy.grain_size == 0) {
		throw std::invalid_argument("CPU policy requires positive threads and grain size");
	}
	const auto size = aView.dimensions();
	const auto count = math::detail::element_count<TView::dimension>(size);
	if (count == 0) {
		return {};
	}
	const auto tasks = 1 + (count - 1) / aPolicy.grain_size;
	const auto workers = std::min(aPolicy.threads, tasks);
	if (workers == 1) {
		typename TView::coord_t position{};
		do {
			aFunction(position);
		} while (math::detail::advance<TView::dimension>(position, size));
		return {};
	}

	std::vector<std::exception_ptr> errors(workers);
	const auto run_worker = [&](std::size_t aWorker, TFunction& aCallback) {
		try {
			const auto base = tasks / workers;
			const auto extra = tasks % workers;
			const auto first = aWorker * base + std::min(aWorker, extra);
			const auto last = first + base + (aWorker < extra);
			for (auto task = first; task < last; ++task) {
				const auto begin = task * aPolicy.grain_size;
				const auto length = std::min(aPolicy.grain_size, count - begin);
				auto position = math::detail::coordinate_at<TView::dimension,
					typename TView::coord_t>(begin, size);
				for (std::size_t i = 0; i < length; ++i) {
					aCallback(position);
					math::detail::advance<TView::dimension>(position, size);
				}
			}
		} catch (...) {
			errors[aWorker] = std::current_exception();
		}
	};
	{
		std::vector<std::jthread> threads;
		threads.reserve(workers - 1);
		for (std::size_t worker = 1; worker < workers; ++worker) {
			threads.emplace_back([&, worker, callback = aFunction]() mutable {
				run_worker(worker, callback);
			});
		}
		run_worker(0, aFunction);
	}
	for (const auto& error : errors) {
		if (error) {
			std::rethrow_exception(error);
		}
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
cuda_execution visit_coordinates(TView aView, TFunction aFunction, cuda_policy aPolicy = {})
{
	const auto size = aView.dimensions();
	const auto count = math::detail::element_count<TView::dimension>(size);
	if (count != 0) {
		constexpr unsigned int threads = 256;
		const auto blocks = 1 + (count - 1) / threads;
		if (blocks > 2147483647u) {
			throw std::invalid_argument("CUDA traversal exceeds grid size limit");
		}
		visit_positions<TView><<<static_cast<unsigned int>(blocks), threads, 0, aPolicy.stream>>>(
			aFunction, count, size);
		detail::check_cuda(cudaGetLastError(), "CUDA traversal launch");
	}
	return {aPolicy.stream};
}
#endif

} // namespace detail

/// Execution policy deduced from a view's backend tag.
template<concepts::AnyView TView>
using view_policy =
#if defined(__CUDACC__)
	std::conditional_t<concepts::DeviceView<TView>, cuda_policy, cpu_policy>;
#else
	cpu_policy;
#endif

/// Completion handle associated with a view's backend.
template<concepts::AnyView TView>
using view_execution =
#if defined(__CUDACC__)
	std::conditional_t<concepts::DeviceView<TView>, cuda_execution, cpu_execution>;
#else
	cpu_execution;
#endif

/**
 * @brief Visit each pixel with its value or reference and coordinate.
 * @param aPolicy Backend-specific scheduling options.
 * @return Execution handle for waiting on completion.
 * @throws std::invalid_argument If the CPU policy has a zero field.
 * @pre For CUDA, the callable must support device execution and storage must
 * outlive the operation. Invocations may run concurrently in any order.
 *
 * CPU callback exceptions propagate after joining workers. Output may be partial.
 */
template<concepts::ViewDomain TView, class TFunction>
auto for_each_position(TView aView, TFunction aFunction, view_policy<TView> aPolicy = {})
{
	return detail::visit_coordinates(aView, detail::pixel_visitor<TView, TFunction>{aView, aFunction}, aPolicy);
}

/// Visit each pixel with its value or reference.
template<concepts::ViewDomain TView, class TFunction>
auto for_each(TView aView, TFunction aFunction, view_policy<TView> aPolicy = {})
{
	return for_each_position(aView, detail::value_visitor<TFunction>{aFunction}, aPolicy);
}

/// Visit each position with a read-only neighborhood locator.
template<concepts::ReadableView TView, class TFunction>
auto for_each_locator(TView aView, TFunction aFunction, border_mode aBorder = border_mode::clamp,
	view_policy<TView> aPolicy = {})
{
	return detail::visit_coordinates(aView, detail::locator_visitor<TView, TFunction>{aView, aFunction, aBorder}, aPolicy);
}

} // namespace iane
