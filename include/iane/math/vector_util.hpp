#pragma once
/**
 * @file vector_util.hpp
 * @brief Utility functions for iane::math::vector (comparisons, norms, rounding).
 */

#include <cmath>
#include <ostream>
#include <type_traits>
#include <utility>

#include <iane/config.hpp>
#include <iane/math/vector.hpp>
#include <iane/math/vector_ops.hpp>

namespace iane::math {

//------------------------------------------------------------
// Coordinate casting
//------------------------------------------------------------

template<typename TOther, typename TElement, int tN>
IANE_DECL_HYBRID constexpr vector<TOther, tN>
coord_cast(const vector<TElement, tN> &aV)
{
	return vector<TOther, tN>(aV);
}

//------------------------------------------------------------
// Relational operators
//------------------------------------------------------------

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr bool
operator==(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	for (int i = 0; i < tN; ++i) {
		if (aFirst.mElements[i] != aSecond.mElements[i]) {
			return false;
		}
	}
	return true;
}

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr bool
operator!=(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	return !(aFirst == aSecond);
}

/**
 * @brief Strict component-wise less-than: a[i] < b[i] for all i.
 */
template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr bool
less(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	for (int i = 0; i < tN; ++i) {
		if (aFirst.mElements[i] >= aSecond.mElements[i]) {
			return false;
		}
	}
	return true;
}

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr bool
operator<(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	return less(aFirst, aSecond);
}

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr bool
operator<=(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	for (int i = 0; i < tN; ++i) {
		if (aFirst.mElements[i] > aSecond.mElements[i]) {
			return false;
		}
	}
	return true;
}

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr bool
operator>=(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	return aSecond <= aFirst;
}

//------------------------------------------------------------
// Element-wise min/max between two vectors
//------------------------------------------------------------

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr auto
max_coords(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	using R = std::common_type_t<TElement, TOther>;
	vector<R, tN> out;
	for (int i = 0; i < tN; ++i) {
		out.mElements[i] = (aFirst.mElements[i] > aSecond.mElements[i]) ? aFirst.mElements[i] : aSecond.mElements[i];
	}
	return out;
}

template<typename TElement, typename TOther, int tN>
IANE_DECL_HYBRID constexpr auto
min_coords(const vector<TElement, tN> &aFirst, const vector<TOther, tN> &aSecond)
{
	using R = std::common_type_t<TElement, TOther>;
	vector<R, tN> out;
	for (int i = 0; i < tN; ++i) {
		out.mElements[i] = (aFirst.mElements[i] < aSecond.mElements[i]) ? aFirst.mElements[i] : aSecond.mElements[i];
	}
	return out;
}

//------------------------------------------------------------
// Stream output (host-only)
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HOST inline std::ostream &
operator<<(std::ostream &aOs, const vector<TElement, tN> &aV)
{
	aOs << "[ ";
	for (int i = 0; i < tN - 1; ++i) {
		aOs << aV[i] << ", ";
	}
	return aOs << aV[tN - 1] << " ]";
}

//------------------------------------------------------------
// Norms
//------------------------------------------------------------

template<MathVector TVector>
IANE_DECL_HYBRID constexpr auto
squared_magnitude(const TVector &aV)
{
	return dot(aV, aV);
}

template<MathVector TVector>
IANE_DECL_HYBRID inline auto
magnitude(const TVector &aV)
{
	using std::sqrt;
	auto sq = squared_magnitude(aV);
#ifdef __CUDA_ARCH__
	// assume float-like for typical coordinates
	return sqrtf(static_cast<float>(sq));
#else
	return sqrt(sq);
#endif
}

template<MathVector TVector>
IANE_DECL_HYBRID inline TVector
normalize(const TVector &aV)
{
	auto mag = magnitude(aV);
	return (1.0f / mag) * aV;
}

//------------------------------------------------------------
// Element-wise product between two vectors
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr vector<TElement, tN>
product(vector<TElement, tN> aFirst, const vector<TElement, tN> &aSecond)
{
	for (int i = 0; i < tN; ++i) {
		aFirst[i] *= aSecond[i];
	}
	return aFirst;
}

//------------------------------------------------------------
// Element-wise abs
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN>
abs(const vector<TElement, tN> &aV)
{
	using std::abs;
	vector<TElement, tN> out;
	for (int i = 0; i < tN; ++i) {
		out[i] = abs(aV[i]);
	}
	return out;
}

//------------------------------------------------------------
// Min/max over components
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline TElement
max(const vector<TElement, tN> &aV)
{
	TElement m = aV[0];
	for (int i = 1; i < tN; ++i) {
		m = m < aV[i] ? aV[i] : m;
	}
	return m;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline TElement
min(const vector<TElement, tN> &aV)
{
	TElement m = aV[0];
	for (int i = 1; i < tN; ++i) {
		m = m > aV[i] ? aV[i] : m;
	}
	return m;
}

//------------------------------------------------------------
// Per-element min / max with another vector
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN>
min_per_element(vector<TElement, tN> aFirst, const vector<TElement, tN> &aSecond)
{
	for (int i = 0; i < tN; ++i) {
		aFirst[i] = (aFirst[i] < aSecond[i]) ? aFirst[i] : aSecond[i];
	}
	return aFirst;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN>
max_per_element(vector<TElement, tN> aFirst, const vector<TElement, tN> &aSecond)
{
	for (int i = 0; i < tN; ++i) {
		aFirst[i] = (aFirst[i] > aSecond[i]) ? aFirst[i] : aSecond[i];
	}
	return aFirst;
}

//------------------------------------------------------------
// Sum of components
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline TElement
sum(const vector<TElement, tN> &aV)
{
	TElement s = aV[0];
	for (int i = 1; i < tN; ++i) {
		s += aV[i];
	}
	return s;
}

//------------------------------------------------------------
// Division utilities
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN>
div(const vector<TElement, tN> &aV, TElement aValue)
{
	vector<TElement, tN> out;
	for (int i = 0; i < tN; ++i) {
		out[i] = aV[i] / aValue;
	}
	return out;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN>
div(const vector<TElement, tN> &aFirst, const vector<TElement, tN> &aSecond)
{
	vector<TElement, tN> out;
	for (int i = 0; i < tN; ++i) {
		out[i] = aFirst[i] / aSecond[i];
	}
	return out;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN>
div_up(const vector<TElement, tN> &aFirst, const vector<TElement, tN> &aSecond)
{
	vector<TElement, tN> out;
	for (int i = 0; i < tN; ++i) {
		out[i] = (aFirst[i] + aSecond[i] - 1) / aSecond[i];
	}
	return out;
}

//------------------------------------------------------------
// Product of all components
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline TElement
product(const vector<TElement, tN> &aV)
{
	TElement r = TElement{1};
	for (int i = 0; i < tN; ++i) {
		r *= aV[i];
	}
	return r;
}

template<typename TElement>
IANE_DECL_HYBRID inline typename std::enable_if<std::is_fundamental<TElement>::value, TElement>::type
product(const TElement &aValue)
{
	return aValue;
}

//------------------------------------------------------------
// Rounding helpers (round, floor, ceil)
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<int, tN>
round(const vector<TElement, tN> &aV)
{
	vector<int, tN> out;
	for (int i = 0; i < tN; ++i) {
#ifdef __CUDA_ARCH__
		out[i] = ::roundf(static_cast<float>(v[i]));
#else
		using std::round;
		out[i] = static_cast<int>(round(aV[i]));
#endif
	}
	return out;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<int, tN>
floor(const vector<TElement, tN> &aV)
{
	vector<int, tN> out;
	for (int i = 0; i < tN; ++i) {
#ifdef __CUDA_ARCH__
		out[i] = ::floorf(static_cast<float>(v[i]));
#else
		using std::floor;
		out[i] = static_cast<int>(floor(aV[i]));
#endif
	}
	return out;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<int, tN>
ceil(const vector<TElement, tN> &aV)
{
	vector<int, tN> out;
	for (int i = 0; i < tN; ++i) {
#ifdef __CUDA_ARCH__
		out[i] = ::ceilf(static_cast<float>(v[i]));
#else
		using std::ceil;
		out[i] = static_cast<int>(ceil(aV[i]));
#endif
	}
	return out;
}

//------------------------------------------------------------
// Insert / remove dimension
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN + 1>
insert_dimension(const vector<TElement, tN> &aV, TElement aValue, int aDimIndex)
{
	IANE_ASSERT(aDimIndex >= 0);
	IANE_ASSERT(aDimIndex <= tN);
	vector<TElement, tN + 1> out;
	for (int i = 0; i < aDimIndex; ++i) {
		out[i] = aV[i];
	}
	out[aDimIndex] = aValue;
	for (int i = aDimIndex; i < tN; ++i) {
		out[i + 1] = aV[i];
	}
	return out;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID inline vector<TElement, tN - 1>
remove_dimension(const vector<TElement, tN> &aV, int aDimIndex)
{
	IANE_ASSERT(aDimIndex >= 0);
	IANE_ASSERT(aDimIndex < tN);
	vector<TElement, tN - 1> out;
	for (int i = 0; i < aDimIndex; ++i) {
		out[i] = aV[i];
	}
	for (int i = aDimIndex; i < tN - 1; ++i) {
		out[i] = aV[i + 1];
	}
	return out;
}

//------------------------------------------------------------
// Sort components (simple bubble sort)
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID inline void
sort(vector<TElement, tN> &aV)
{
	for (int i = 0; i < tN; ++i) {
		for (int j = 1; j < tN - i; ++j) {
			if (aV[j] < aV[j - 1]) {
				using std::swap;
				swap(aV[j], aV[j - 1]);
			}
		}
	}
}

//------------------------------------------------------------
// Find an orthonormal vector to a 3D vector
//------------------------------------------------------------

template<MathVector TVector>
IANE_DECL_HYBRID inline auto
find_orthonormal(const TVector &aV)
{
	static_assert(TVector::dim == 3, "find_orthonormal() requires 3D vector");
	using T = typename TVector::value_type;
	vector<T, 3> result{};
	for (int j = 0; j < 3; ++j) {
		if (aV[j] != T{}) {
			auto x = aV[j];
			auto y = aV[(j + 1) % 3];
			auto norm = 1.0 / std::sqrt(x * x + y * y);
			result[j] = static_cast<T>(y * norm);
			result[(j + 1) % 3] = static_cast<T>(-x * norm);
			result[(j + 2) % 3] = T{};
			return result;
		}
	}
	return result;
}

//------------------------------------------------------------
// 2D rotation
//------------------------------------------------------------

template<typename TElement>
IANE_DECL_HYBRID inline vector<TElement, 2>
rotate(const vector<TElement, 2> &aV, double aAngle)
{
	using std::sin;
	using std::cos;
	auto sa = sin(aAngle);
	auto ca = cos(aAngle);

	return vector<TElement, 2>{
		static_cast<TElement>(ca * aV[0] - sa * aV[1]),
		static_cast<TElement>(sa * aV[0] + ca * aV[1])
	};
}

} // namespace iane::math
