#pragma once

#include <concepts>

namespace iane::concepts {

/**
 * @defgroup iane_concepts IANE concepts
 * @brief Compile-time contracts for images, views, coordinates and elements.
 *
 * Include <iane/concepts.hpp> for the full set, or include the individual
 * headers under iane/concepts. Concepts check expressions and type relationships.
 * Runtime properties such as extent equality, valid storage, and GPU
 * availability are checked or documented by the consuming operation.
 *
 * Start with @ref iane::concepts::ReadableView "ReadableView" for algorithms that only sample coordinates.
 * Use @ref iane::concepts::TransformViews "TransformViews" to pair a source with a writable destination, or
 * @ref iane::concepts::TransferViews "TransferViews" when the operation moves bytes across memory spaces.
 * Constrain neighborhood callbacks with @ref iane::concepts::Locator "Locator" and describe pixel
 * arithmetic separately with an element concept.
 */

/**
 * @ingroup iane_concepts
 * @brief A vector-like type exposing a component type, a dimension, and indexing.
 * @tparam TVector Candidate vector type.
 *
 * Intended for generic coordinate and vector utilities. Component arithmetic
 * is a separate requirement of each operation.
 */
template<class TVector>
concept MathVector = requires(TVector& aVector) {
	typename TVector::value_type;
	{ TVector::dim } -> std::convertible_to<int>;
	aVector[0];
};

} // namespace iane::concepts
