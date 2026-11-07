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

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr auto
operator+(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	using R = std::common_type_t<TElement, TOther>;
	vector<R, tN> out(aFirst);
	out += aSecond;
	return out;
}

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr auto
operator-(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	using R = std::common_type_t<TElement, TOther>;
	vector<R, tN> out(aFirst);
	out -= aSecond;
	return out;
}

//------------------------------------------------------------
// Scalar multiplication
//------------------------------------------------------------

template<typename TScalar, typename TElement, int tN>
IANE_DECL_HYBRID constexpr auto
operator*(TScalar aScalar, const vector<TElement, tN> &aV)
{
	using R = decltype(aScalar * aV[0]);
	vector<R, tN> out;
	for (int i = 0; i < tN; ++i) {
		out[i] = aScalar * aV[i];
	}
	return out;
}

template<typename TScalar, typename TElement, int tN>
IANE_DECL_HYBRID constexpr auto
operator*(const vector<TElement, tN> &aV, TScalar aScalar)
{
	return aScalar * aV;
}

//------------------------------------------------------------
// Dot product
//------------------------------------------------------------

template<MathVector TFirstVector, MathVector TSecondVector>
IANE_DECL_HYBRID constexpr auto
dot(const TFirstVector &aFirst, const TSecondVector &aSecond)
{
	static_assert(TFirstVector::dim == TSecondVector::dim, "dot(): dimension mismatch");
	auto r = aFirst[0] * aSecond[0];
	for (int i = 1; i < TFirstVector::dim; ++i) {
		r += aFirst[i] * aSecond[i];
	}
	return r;
}

//------------------------------------------------------------
// Cross product (3D only)
//------------------------------------------------------------

template<MathVector TFirstVector, MathVector TSecondVector>
IANE_DECL_HYBRID constexpr auto
cross(const TFirstVector &aFirst, const TSecondVector &aSecond)
{
	static_assert(TFirstVector::dim == 3 && TSecondVector::dim == 3, "cross() requires 3D vectors");
	using R = decltype(aFirst[0] * aSecond[0]);
	vector<R, 3> out;

	out[0] = aFirst[1] * aSecond[2] - aFirst[2] * aSecond[1];
	out[1] = aFirst[2] * aSecond[0] - aFirst[0] * aSecond[2];
	out[2] = aFirst[0] * aSecond[1] - aFirst[1] * aSecond[0];
	return out;
}

} // namespace iane::math
