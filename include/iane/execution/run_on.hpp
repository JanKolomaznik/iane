#pragma once

#include <iane/algorithms/copy.hpp>

#if defined(__CUDACC__)
#include <iane/resource/cuda/resource_cuda.hpp>
#endif

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace iane {

/// Backend selected at runtime by run_on.
enum class backend { cpu, cuda };

/// Execution and measurement options for run_on.
struct run_options {
	iane::backend backend = backend::cpu;
	int device = 0;
	int warmup = 0;
	int runs = 1;
};

namespace detail {

template<class TOperation>
double elapsed_ms(TOperation&& aOperation)
{
	const auto start = std::chrono::steady_clock::now();
	aOperation();
	return std::chrono::duration<double, std::milli>(
		std::chrono::steady_clock::now() - start).count();
}

template<class TOperation>
void run_samples(const run_options& aOptions, TOperation&& aOperation)
{
	for (int i = 0; i < aOptions.warmup; ++i) {
		aOperation();
	}
	std::vector<double> samples;
	samples.reserve(aOptions.runs);
	for (int i = 0; i < aOptions.runs; ++i) {
		samples.push_back(elapsed_ms(aOperation));
	}
	std::sort(samples.begin(), samples.end());
	const auto middle = samples.size() / 2;
	const auto median = std::midpoint(samples[(samples.size() - 1) / 2], samples[middle]);
	std::cout << "compute_ms min=" << samples.front() << " median=" << median
		<< " max=" << samples.back() << '\n';
}

#if defined(__CUDACC__)
struct device_guard {
	int previous{};

	explicit device_guard(int aDevice)
	{
		check_cuda(cudaGetDevice(&previous), "Get current CUDA device");
		check_cuda(cudaSetDevice(aDevice), "Select CUDA device");
	}

	~device_guard() { cudaSetDevice(previous); }
};

struct stream_owner {
	cudaStream_t stream{};
	~stream_owner()
	{
		if (stream) {
			cudaStreamSynchronize(stream);
			cudaStreamDestroy(stream);
		}
	}
};
#endif

template<bool tPrepare, concepts::HostView TDst, class TOperation, concepts::HostView... TSrc>
requires (((TSrc::dimension > 0) && ...) && TDst::dimension > 0)
void run_on_impl(const run_options& aOptions, std::tuple<TSrc...>& aSources, TDst aDestination, TOperation&& aOperation)
{
	if (aOptions.warmup < 0 || aOptions.runs <= 0 || aOptions.device < 0) {
		throw std::invalid_argument("Invalid run options");
	}

	const auto domain = [&] {
		if constexpr (sizeof...(TSrc) == 0) {
			return aDestination;
		} else {
			return std::get<0>(aSources);
		}
	}();
	const auto size = domain.dimensions();
	const bool use_cuda = aOptions.backend == backend::cuda;
	std::cout << "backend=" << (use_cuda ? "cuda" : "cpu");
	if (use_cuda) {
		std::cout << " device=" << aOptions.device;
	}
	std::cout << " size=" << math::detail::axis(size, 0);
	constexpr auto dimension = std::remove_cvref_t<decltype(domain)>::dimension;
	for (int i = 1; i < dimension; ++i) {
		std::cout << 'x' << math::detail::axis(size, i);
	}
	std::cout << " warmup=" << aOptions.warmup << " runs=" << aOptions.runs << '\n';

	const auto run = [&](auto& aExecution, auto aInvoke) {
		if constexpr (tPrepare) {
			const auto start = std::chrono::steady_clock::now();
			auto operation = aInvoke();
			aExecution.wait();
			const auto elapsed = std::chrono::duration<double, std::milli>(
				std::chrono::steady_clock::now() - start).count();
			std::cout << "prepare_ms=" << elapsed << '\n';
			detail::run_samples(aOptions, [&] { operation().wait(); });
		} else {
			detail::run_samples(aOptions, [&] {
				aInvoke().wait();
			});
		}
	};

	if (!use_cuda) {
		cpu_execution execution;
		std::apply([&](const auto&... aInput) {
			run(execution, [&]() -> decltype(auto) {
				return aOperation(aInput..., aDestination, execution);
			});
		}, aSources);
		return;
	}

#if defined(__CUDACC__)
	std::optional<detail::device_guard> device;
	std::tuple<device_image<typename TSrc::value_type, TSrc::dimension>...> inputs;
	device_image<typename TDst::value_type, TDst::dimension> output;
	detail::stream_owner queue;
	const auto setup = detail::elapsed_ms([&] {
		device.emplace(aOptions.device);
		detail::check_cuda(cudaStreamCreateWithFlags(&queue.stream, cudaStreamNonBlocking),
			"Create CUDA stream");
		[&]<std::size_t... tI>(std::index_sequence<tI...>) {
			(std::get<tI>(inputs).resize(std::get<tI>(aSources).dimensions()), ...);
		}(std::index_sequence_for<TSrc...>{});
		output.resize(aDestination.dimensions());
	});
	cuda_execution execution{queue.stream};
	std::cout << "setup_ms=" << setup;
	if constexpr (sizeof...(TSrc) > 0) {
		const auto upload = detail::elapsed_ms([&] {
			[&]<std::size_t... tI>(std::index_sequence<tI...>) {
				(copy(std::get<tI>(aSources), std::get<tI>(inputs).view(), execution), ...);
			}(std::index_sequence_for<TSrc...>{});
			execution.wait();
		});
		std::cout << " upload_ms=" << upload;
	}
	std::cout << '\n';
	std::apply([&](const auto&... aInput) {
		run(execution, [&]() -> decltype(auto) {
			return aOperation(aInput.const_view()..., output.view(), execution);
		});
	}, inputs);
	const auto download = detail::elapsed_ms([&] {
		copy(output.const_view(), aDestination, execution).wait();
	});
	std::cout << "download_ms=" << download << '\n';
#else
	throw std::runtime_error("CUDA execution requires compilation with a CUDA compiler");
#endif
}

} // namespace detail

/**
 * @brief Run an operation on host views or CUDA copies of those views.
 * @param aOptions Backend, device and measurement settings.
 * @param aSources Tuple of readable host memory views, optionally empty for generators.
 * @param aDestination Writable host memory view.
 * @param aOperation Callable taking the unpacked input views, output view and execution
 * context. It returns a result supporting wait() and must produce the output
 * on each invocation.
 * @throws std::invalid_argument If options are invalid.
 * @throws std::runtime_error If CUDA is unavailable or a CUDA operation fails.
 *
 * Logs host wall-clock timings to standard output. The destination is ready
 * on return. The CUDA path stages input once and downloads the final output.
 */
template<concepts::HostView TDst, class TOperation, concepts::HostView... TSrc>
requires (((TSrc::dimension > 0) && ...) && TDst::dimension > 0)
void run_on(const run_options& aOptions, std::tuple<TSrc...> aSources, TDst aDestination, TOperation&& aOperation)
{
	detail::run_on_impl<false>(aOptions, aSources, aDestination, std::forward<TOperation>(aOperation));
}

/**
 * @brief Prepare a reusable operation after backend setup and input staging.
 * @param aPrepare Factory taking the backend views and execution context.
 * It returns a nullary callable that produces the complete output and returns
 * a completion handle each time. Factory work must use the supplied context.
 * The returned callable should retain view handles by value.
 *
 * The factory runs once. Its elapsed time through completion is logged
 * as prepare_ms, separately from warm-up, compute samples and transfers.
 * The callable is destroyed before the output download. See run_on for
 * backend selection and the other arguments.
 */
template<concepts::HostView TDst, class TPrepare, concepts::HostView... TSrc>
requires (((TSrc::dimension > 0) && ...) && TDst::dimension > 0)
void run_prepared_on(const run_options& aOptions, std::tuple<TSrc...> aSources,
	TDst aDestination, TPrepare&& aPrepare)
{
	detail::run_on_impl<true>(aOptions, aSources, aDestination, std::forward<TPrepare>(aPrepare));
}

/// Run a single-input operation. See the tuple overload for execution and timing.
template<concepts::HostView TSrc, concepts::HostView TDst, class TOperation>
requires (TSrc::dimension > 0 && TDst::dimension > 0)
void run_on(const run_options& aOptions, TSrc aSource, TDst aDestination, TOperation&& aOperation)
{
	run_on(aOptions, std::tuple{aSource}, aDestination, std::forward<TOperation>(aOperation));
}

/// Generate an output without staging any input. The callback takes the output and execution context.
template<concepts::HostView TDst, class TOperation>
requires (TDst::dimension > 0)
void run_on(const run_options& aOptions, TDst aDestination, TOperation&& aOperation)
{
	run_on(aOptions, std::tuple{}, aDestination, std::forward<TOperation>(aOperation));
}

} // namespace iane
