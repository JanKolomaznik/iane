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

#include <iane/config.hpp> // for IANE_DECL_HOST / _DEVICE / _HYBRID

namespace iane::math {

/**
 * @brief Concept for vector-like types usable with IANE utilities.
 */
template<typename T>
concept MathVector = requires(T v) {
	typename T::value_type;
	{ T::dim } -> std::convertible_to<int>;
	{ v[0] };
};

/**
 * @brief N-dimensional vector with element type T.
 *
 * Lightweight POD-like fixed-size vector usable on host and device.
 *
 * @tparam T Element type
 * @tparam N Dimension (positive integer)
 */
template<typename T, int N>
class vector {
public:
	using value_type = T;
	static constexpr int dim = N;

	/// Storage
	T elems[N];

	/// @name Constructors
	/// @{

	/// Default initializes all components to zero.
	IANE_DECL_HYBRID constexpr vector();

	/// Fill constructor: sets all elements to `value`.
	IANE_DECL_HYBRID constexpr vector(T value, std::true_type);

	/// Construct from a list of explicit coordinates.
	IANE_DECL_HYBRID explicit constexpr vector(std::initializer_list<T> init);

	/// Construct from another vector of convertible type.
	template<typename U>
	IANE_DECL_HYBRID constexpr vector(const vector<U, N> &other);

	/// @}

	/// @name Element Access
	/// @{

	IANE_DECL_HYBRID constexpr T &operator[](int idx);
	IANE_DECL_HYBRID constexpr const T &operator[](int idx) const;

	/// Access a compile-time index.
	template<int I>
	IANE_DECL_HYBRID constexpr T &get();

	template<int I>
	IANE_DECL_HYBRID constexpr const T &get() const;

	/// @}

	/// @name Arithmetic assignment
	/// @{

	template<typename U>
	IANE_DECL_HYBRID constexpr vector &operator+=(const vector<U, N> &rhs);

	template<typename U>
	IANE_DECL_HYBRID constexpr vector &operator-=(const vector<U, N> &rhs);

	/// Unary minus
	IANE_DECL_HYBRID constexpr vector operator-() const;

	/// @}
};

} // namespace iane::math

#include <iane/math/detail/vector.tcc>
