#pragma once

/**
 * @file config.hpp
 * @brief CUDA-specific qualifiers and configuration macros for IANE.
 *
 * This header is included only when compiling with NVCC or Clang CUDA mode.
 * It provides CUDA function qualifiers and assertion macros, isolated in
 * the `iane::cuda` namespace to avoid macro leakage into the global scope.
 *
 * @ingroup iane_resource_cuda
 */

#include <cassert>
#include <cuda_runtime.h>

namespace iane::cuda {

/// Indicates CUDA support is enabled at compile time.
inline constexpr bool available = true;

/// CUDA function qualifiers (namespaced macro wrappers)
#define IANE_CUDA_HOST		__host__
#define IANE_CUDA_DEVICE	__device__
#define IANE_CUDA_HYBRID	__host__ __device__
#define IANE_CUDA_GLOBAL	__global__
#define IANE_CUDA_CONST		__constant__
#define IANE_CUDA_SHARED	__shared__

/// Disable host-call warnings inside device code (NVCC extension).
#define IANE_CUDA_HD_WARNING_DISABLE	_Pragma("hd_warning_disable")

/**
 * @brief Assertion wrapper for host or device code.
 *
 * Device-side assertions only trigger when compiled with debug info
 * (-G flag). Otherwise, they behave as no-ops.
 */
#define IANE_CUDA_ASSERT(EXPR)	assert((EXPR))

/**
 * @brief Assert that a CUDA runtime call returns success.
 *
 * Example:
 * @code
 * IANE_CUDA_ASSERT_SUCCESS(cudaMalloc(&ptr, size));
 * @endcode
 */
#define IANE_CUDA_ASSERT_SUCCESS(EXPR)	\
	IANE_CUDA_ASSERT((EXPR) == cudaSuccess)

} // namespace iane::cuda
