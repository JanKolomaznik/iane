#pragma once
/**
 * @file vector.hpp
 * @brief Fixed-size N-dimensional mathematical vector.
 *
 * Declares the iane::math::vector template used for coordinates
 * and mathematical operations. Template bodies are located in
 * detail/vector.tcc.
 */

#include <cstddef>
#include <initializer_list>
#include <type_traits>
#include <concepts>

#include <iane/config.hpp> // for IANE_DECL_HOST / _DEVICE / _HYBRID

namespace iane::math {

/**
 * @brief Concept for vector-like types usable with IANE utilities.
 */
template<typename TElement>
concept MathVector = requires(TElement aV) {
	typename TElement::value_type;
	{ TElement::dim } -> std::convertible_to<int>;
	{ aV[0] };
};

/**
 * @brief N-dimensional vector with element type T.
 *
 * Lightweight POD-like fixed-size vector usable on host and device.
 *
 * @tparam TElement Element type
 * @tparam tN Dimension (positive integer)
 */
template<typename TElement, int tN>
class vector {
public:
	using value_type = TElement;
	static constexpr int dim = tN;

	/// Storage
	TElement mElements[tN];

	/// @name Constructors
	/// @{

	/// Default initializes all components to zero.
	IANE_DECL_HYBRID constexpr vector();

	/// Fill constructor: sets all elements to `aValue`.
	IANE_DECL_HYBRID constexpr vector(TElement aValue, std::true_type);

	/// Construct from a list of explicit coordinates.
	IANE_DECL_HYBRID explicit constexpr vector(std::initializer_list<TElement> aInit);

	/// Construct from another vector of convertible type.
	template<typename TOther>
	IANE_DECL_HYBRID constexpr vector(const vector<TOther, tN> &aOther);

	/// @}

	/// @name Element Access
	/// @{

	IANE_DECL_HYBRID constexpr TElement &operator[](int aIdx);
	IANE_DECL_HYBRID constexpr const TElement &operator[](int aIdx) const;

	/// Access a compile-time index.
	template<int tI>
	IANE_DECL_HYBRID constexpr TElement &get();

	template<int tI>
	IANE_DECL_HYBRID constexpr const TElement &get() const;

	/// @}

	/// @name Arithmetic assignment
	/// @{

	template<typename TOther>
	IANE_DECL_HYBRID constexpr vector &operator+=(const vector<TOther, tN> &aRhs);

	template<typename TOther>
	IANE_DECL_HYBRID constexpr vector &operator-=(const vector<TOther, tN> &aRhs);

	/// Unary minus
	IANE_DECL_HYBRID constexpr vector operator-() const;

	/// @}
};

} // namespace iane::math

#include <iane/math/detail/vector.tcc>
