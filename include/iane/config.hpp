#pragma once
/**
 * @file config.hpp
 * @brief Global configuration for the IANE library (backend independent).
 */

#include <cassert>

/// Assertion macro used across IANE.
#define IANE_ASSERT(EXPR)	assert((EXPR))

/// General-purpose force inline
#define IANE_FORCE_INLINE	inline


// -----------------------------------------------------------------------------
// CUDA support (fallback + activation)
// Centralized in resource/cuda.hpp
// -----------------------------------------------------------------------------

#include <iane/resource/cuda.hpp>
