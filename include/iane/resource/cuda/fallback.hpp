#pragma once

/**
 * @file fallback.hpp
 * @brief Fallback definitions when CUDA support is not available.
 *
 * These definitions allow code to compile even when CUDA is not present.
 * All CUDA-specific qualifiers expand to nothing, and CUDA types are
 * replaced with lightweight stand-ins that prevent accidental misuse.
 *
 * @ingroup iane_resource_cuda
 */

#include <cstddef>

namespace iane::cuda {

/// Indicates CUDA support is not available.
inline constexpr bool available = false;

/// Empty qualifiers in non-CUDA builds
#define IANE_CUDA_HOST
#define IANE_CUDA_DEVICE
#define IANE_CUDA_HYBRID
#define IANE_CUDA_GLOBAL
#define IANE_CUDA_CONST
#define IANE_CUDA_SHARED
#define IANE_CUDA_HD_WARNING_DISABLE

/// Opaque fallback stream type (safer than using size_t)
struct fake_stream_t {};
using cudaStream_t = fake_stream_t;

/// No-op assertions
#define IANE_CUDA_ASSERT(EXPR)			do { (void)(EXPR); } while (0)
#define IANE_CUDA_ASSERT_SUCCESS(EXPR)	do { (void)(EXPR); } while (0)

} // namespace iane::cuda
