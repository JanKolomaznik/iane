#pragma once
/**
 * @file exception.hpp
 * @brief CUDA-specific error handling for IANE.
 *
 * Included only when CUDA support is active.
 */

#include <iane/resource/cuda/config.hpp>
#include <iane/resource/cuda/error_codes.hpp>
#include <iane/logging.hpp>
#include <iane/exception.hpp>	// for ExceptionBase + MessageInfo

#include <boost/exception/all.hpp>
#include <cuda_runtime.h>

namespace iane::cuda {

/// Additional diagnostic info storing a CUDA error code.
using CudaErrorInfo = boost::error_info<struct tag_cuda_error_code, cudaError_t>;

} // namespace iane::cuda

/**
 * @brief Check CUDA call and throw ExceptionBase on failure.
 *
 * Example:
 *	IANE_CHECK_CUDA(cudaMalloc(&ptr, size));
 */
#define IANE_CHECK_CUDA_MSG(msg, expr)												\
	do {																			\
		cudaError_t err = (expr);													\
		if (err != cudaSuccess) {													\
			IANE_EFORMAT("%1% (%2%) %3%",											\
				(msg),																\
				iane::cuda::get_error_name(err),									\
				cudaGetErrorString(err));											\
			IANE_THROW(iane::ExceptionBase()										\
				<< iane::MessageInfo(msg)											\
				<< iane::cuda::CudaErrorInfo(err));									\
		}																			\
	} while(false)

/// Use expression as message
#define IANE_CHECK_CUDA(expr)			IANE_CHECK_CUDA_MSG(#expr, expr)

/// Check global CUDA error state
#define IANE_CHECK_CUDA_STATE(msg)		IANE_CHECK_CUDA_MSG(msg, cudaGetLastError())
