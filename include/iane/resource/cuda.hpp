#pragma once

/**
 * @defgroup iane_resource_cuda CUDA resources
 * @brief CUDA execution support, resource configuration, and diagnostics.
 */

/**
 * @file cuda.hpp
 * @brief Unified include header for CUDA-related configuration.
 *
 * This header selects either CUDA runtime support or the fallback version,
 * depending on the active compiler and build configuration.
 *
 * @ingroup iane_resource_cuda
 */

#if defined(__CUDACC__) || defined(__CUDA_ARCH__) || defined(__clang_cuda__)
	#include <iane/resource/cuda/config.hpp>
#else
	#include <iane/resource/cuda/fallback.hpp>
#endif

/// TODO: remove?
#define IANE_DECL_HYBRID IANE_CUDA_HYBRID
#define IANE_DECL_HOST IANE_CUDA_HOST
