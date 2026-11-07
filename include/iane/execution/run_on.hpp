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

} // namespace detail

/**
 * @brief Run an operation on host views or CUDA copies of those views.
 * @param aOptions Backend, device and measurement settings.
 * @param aSources Tuple of readable host memory views.
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
requires (sizeof...(TSrc) > 0 && ((TSrc::dimension > 0) && ...) && TDst::dimension > 0)
void run_on(const run_options& aOptions, std::tuple<TSrc...> aSources, TDst aDestination, TOperation&& aOperation)
{
	if (aOptions.warmup < 0 || aOptions.runs <= 0 || aOptions.device < 0) {
		throw std::invalid_argument("Invalid run options");
	}

	const auto size = std::get<0>(aSources).dimensions();
	const bool use_cuda = aOptions.backend == backend::cuda;
	std::cout << "backend=" << (use_cuda ? "cuda" : "cpu");
	if (use_cuda) {
		std::cout << " device=" << aOptions.device;
	}
	std::cout << " size=" << math::detail::axis(size, 0);
	constexpr auto dimension = std::tuple_element_t<0, std::tuple<TSrc...>>::dimension;
	for (int i = 1; i < dimension; ++i) {
		std::cout << 'x' << math::detail::axis(size, i);
	}
	std::cout << " warmup=" << aOptions.warmup << " runs=" << aOptions.runs << '\n';

	if (!use_cuda) {
		cpu_execution execution;
		detail::run_samples(aOptions, [&] {
			std::apply([&](const auto&... aInput) {
				aOperation(aInput..., aDestination, execution).wait();
			}, aSources);
		});
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
	const auto upload = detail::elapsed_ms([&] {
		[&]<std::size_t... tI>(std::index_sequence<tI...>) {
			(copy(std::get<tI>(aSources), std::get<tI>(inputs).view(), execution), ...);
		}(std::index_sequence_for<TSrc...>{});
		execution.wait();
	});
	std::cout << "setup_ms=" << setup << " upload_ms=" << upload << '\n';
	detail::run_samples(aOptions, [&] {
		std::apply([&](const auto&... aInput) {
			aOperation(aInput.const_view()..., output.view(), execution).wait();
		}, inputs);
	});
	const auto download = detail::elapsed_ms([&] {
		copy(output.const_view(), aDestination, execution).wait();
	});
	std::cout << "download_ms=" << download << '\n';
#else
	throw std::runtime_error("CUDA execution requires compilation with a CUDA compiler");
#endif
}

/// Run a single-input operation. See the tuple overload for execution and timing.
template<concepts::HostView TSrc, concepts::HostView TDst, class TOperation>
requires (TSrc::dimension > 0 && TDst::dimension > 0)
void run_on(const run_options& aOptions, TSrc aSource, TDst aDestination, TOperation&& aOperation)
{
	run_on(aOptions, std::tuple{aSource}, aDestination, std::forward<TOperation>(aOperation));
}

} // namespace iane
