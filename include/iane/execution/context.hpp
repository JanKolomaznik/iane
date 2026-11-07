#pragma once

#include <iane/config.hpp>

#if defined(__CUDACC__)
#include <stdexcept>
#include <string>
#endif

namespace iane {

/// Synchronous host execution and its already-completed result.
struct cpu_execution {
	void wait() const noexcept {}
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

/// Non-owning CUDA stream handle. The stream must outlive submitted work.
struct cuda_execution {
	cudaStream_t stream{};

	/// Wait for all preceding work on the stream and report CUDA errors.
	void wait() const { detail::check_cuda(cudaStreamSynchronize(stream), "CUDA stream wait"); }
};
#endif

} // namespace iane
