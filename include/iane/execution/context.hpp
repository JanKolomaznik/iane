#pragma once

#include <iane/config.hpp>

#include <cstddef>

#if defined(__CUDACC__)
#include <stdexcept>
#include <string>
#endif

namespace iane {

/**
 * @brief Run host traversal synchronously with at most the requested workers.
 * @pre Both fields must be positive. A single worker preserves traversal order.
 * Parallel invocations may run concurrently in any order. Each worker has
 * its own callable copy. Any shared state accessed by the callable must be safe.
 */
struct cpu_policy {
	std::size_t threads = 1; ///< Maximum number of workers, including the calling thread.
	std::size_t grain_size = 4096; ///< Maximum number of coordinates per task.
};

/// Synchronous host execution and its already-completed result.
struct cpu_execution {
	void wait() const noexcept {}

	/// Use the default serial policy when passing this legacy context as input.
	operator cpu_policy() const noexcept { return {}; }
};

#if defined(__CUDACC__)
namespace detail {

inline void check_cuda(cudaError_t aStatus, const char* aOperation)
{
	if (aStatus != cudaSuccess) {
		throw std::runtime_error(std::string(aOperation) + ": " + cudaGetErrorString(aStatus));
	}
}

} // namespace detail

/// Queue CUDA work on a borrowed stream that must outlive submitted work.
struct cuda_policy {
	cudaStream_t stream{};
};

/// Non-owning CUDA stream handle. The stream must outlive submitted work.
struct cuda_execution {
	cudaStream_t stream{};

	/// Use the same stream when passing this legacy context as input.
	operator cuda_policy() const noexcept { return {stream}; }

	/// Wait for all preceding work on the stream and report CUDA errors.
	void wait() const { detail::check_cuda(cudaStreamSynchronize(stream), "CUDA stream wait"); }
};
#endif

} // namespace iane
