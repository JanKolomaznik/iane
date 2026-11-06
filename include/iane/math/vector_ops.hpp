#pragma once
/**
 * @file vector_ops.hpp
 * @brief Arithmetic operations for iane::math::vector.
 */

#include <iane/math/vector.hpp>
#include <type_traits>

namespace iane::math {

//------------------------------------------------------------
// Binary + and -
//------------------------------------------------------------

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr auto
operator+(const vector<T, N> &a, const vector<U, N> &b)
{
	using R = std::common_type_t<T, U>;
	vector<R, N> out(a);
	out += b;
	return out;
}

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr auto
operator-(const vector<T, N> &a, const vector<U, N> &b)
{
	using R = std::common_type_t<T, U>;
	vector<R, N> out(a);
	out -= b;
	return out;
}

//------------------------------------------------------------
// Scalar multiplication
//------------------------------------------------------------

template<typename S, typename T, int N>
IANE_DECL_HYBRID constexpr auto
operator*(S scalar, const vector<T, N> &v)
{
	using R = decltype(scalar * v[0]);
	vector<R, N> out;
	for (int i = 0; i < N; ++i) {
		out[i] = scalar * v[i];
	}
	return out;
}

template<typename S, typename T, int N>
IANE_DECL_HYBRID constexpr auto
operator*(const vector<T, N> &v, S scalar)
{
	return scalar * v;
}

//------------------------------------------------------------
// Dot product
//------------------------------------------------------------

template<MathVector V1, MathVector V2>
IANE_DECL_HYBRID constexpr auto
dot(const V1 &a, const V2 &b)
{
	static_assert(V1::dim == V2::dim, "dot(): dimension mismatch");
	auto r = a[0] * b[0];
	for (int i = 1; i < V1::dim; ++i) {
		r += a[i] * b[i];
	}
	return r;
}

//------------------------------------------------------------
// Cross product (3D only)
//------------------------------------------------------------

template<MathVector V1, MathVector V2>
IANE_DECL_HYBRID constexpr auto
cross(const V1 &a, const V2 &b)
{
	static_assert(V1::dim == 3 && V2::dim == 3, "cross() requires 3D vectors");
	using R = decltype(a[0] * b[0]);
	vector<R, 3> out;

	out[0] = a[1] * b[2] - a[2] * b[1];
	out[1] = a[2] * b[0] - a[0] * b[2];
	out[2] = a[0] * b[1] - a[1] * b[0];
	return out;
}

} // namespace iane::math
