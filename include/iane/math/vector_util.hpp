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

template<typename U, typename T, int N>
IANE_DECL_HYBRID constexpr vector<U, N>
coord_cast(const vector<T, N> &v)
{
	return vector<U, N>(v);
}

//------------------------------------------------------------
// Relational operators
//------------------------------------------------------------

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr bool
operator==(const vector<T, N> &a, const vector<U, N> &b)
{
	for (int i = 0; i < N; ++i) {
		if (a.elems[i] != b.elems[i]) {
			return false;
		}
	}
	return true;
}

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr bool
operator!=(const vector<T, N> &a, const vector<U, N> &b)
{
	return !(a == b);
}

/**
 * @brief Strict component-wise less-than: a[i] < b[i] for all i.
 */
template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr bool
less(const vector<T, N> &a, const vector<U, N> &b)
{
	for (int i = 0; i < N; ++i) {
		if (a.elems[i] >= b.elems[i]) {
			return false;
		}
	}
	return true;
}

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr bool
operator<(const vector<T, N> &a, const vector<U, N> &b)
{
	return less(a, b);
}

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr bool
operator<=(const vector<T, N> &a, const vector<U, N> &b)
{
	for (int i = 0; i < N; ++i) {
		if (a.elems[i] > b.elems[i]) {
			return false;
		}
	}
	return true;
}

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr bool
operator>=(const vector<T, N> &a, const vector<U, N> &b)
{
	return b <= a;
}

//------------------------------------------------------------
// Element-wise min/max between two vectors
//------------------------------------------------------------

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr auto
max_coords(const vector<T, N> &a, const vector<U, N> &b)
{
	using R = std::common_type_t<T, U>;
	vector<R, N> out;
	for (int i = 0; i < N; ++i) {
		out.elems[i] = (a.elems[i] > b.elems[i]) ? a.elems[i] : b.elems[i];
	}
	return out;
}

template<typename T, typename U, int N>
IANE_DECL_HYBRID constexpr auto
min_coords(const vector<T, N> &a, const vector<U, N> &b)
{
	using R = std::common_type_t<T, U>;
	vector<R, N> out;
	for (int i = 0; i < N; ++i) {
		out.elems[i] = (a.elems[i] < b.elems[i]) ? a.elems[i] : b.elems[i];
	}
	return out;
}

//------------------------------------------------------------
// Stream output (host-only)
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HOST inline std::ostream &
operator<<(std::ostream &os, const vector<T, N> &v)
{
	os << "[ ";
	for (int i = 0; i < N - 1; ++i) {
		os << v[i] << ", ";
	}
	return os << v[N - 1] << " ]";
}

//------------------------------------------------------------
// Norms
//------------------------------------------------------------

template<MathVector V>
IANE_DECL_HYBRID constexpr auto
squared_magnitude(const V &v)
{
	return dot(v, v);
}

template<MathVector V>
IANE_DECL_HYBRID inline auto
magnitude(const V &v)
{
	using std::sqrt;
	auto sq = squared_magnitude(v);
#ifdef __CUDA_ARCH__
	// assume float-like for typical coordinates
	return sqrtf(static_cast<float>(sq));
#else
	return sqrt(sq);
#endif
}

template<MathVector V>
IANE_DECL_HYBRID inline V
normalize(const V &v)
{
	auto mag = magnitude(v);
	return (1.0f / mag) * v;
}

//------------------------------------------------------------
// Element-wise product between two vectors
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID constexpr vector<T, N>
product(vector<T, N> a, const vector<T, N> &b)
{
	for (int i = 0; i < N; ++i) {
		a[i] *= b[i];
	}
	return a;
}

//------------------------------------------------------------
// Element-wise abs
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N>
abs(const vector<T, N> &v)
{
	using std::abs;
	vector<T, N> out;
	for (int i = 0; i < N; ++i) {
		out[i] = abs(v[i]);
	}
	return out;
}

//------------------------------------------------------------
// Min/max over components
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline T
max(const vector<T, N> &v)
{
	T m = v[0];
	for (int i = 1; i < N; ++i) {
		m = m < v[i] ? v[i] : m;
	}
	return m;
}

template<typename T, int N>
IANE_DECL_HYBRID inline T
min(const vector<T, N> &v)
{
	T m = v[0];
	for (int i = 1; i < N; ++i) {
		m = m > v[i] ? v[i] : m;
	}
	return m;
}

//------------------------------------------------------------
// Per-element min / max with another vector
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N>
min_per_element(vector<T, N> a, const vector<T, N> &b)
{
	for (int i = 0; i < N; ++i) {
		a[i] = (a[i] < b[i]) ? a[i] : b[i];
	}
	return a;
}

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N>
max_per_element(vector<T, N> a, const vector<T, N> &b)
{
	for (int i = 0; i < N; ++i) {
		a[i] = (a[i] > b[i]) ? a[i] : b[i];
	}
	return a;
}

//------------------------------------------------------------
// Sum of components
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline T
sum(const vector<T, N> &v)
{
	T s = v[0];
	for (int i = 1; i < N; ++i) {
		s += v[i];
	}
	return s;
}

//------------------------------------------------------------
// Division utilities
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N>
div(const vector<T, N> &v, T value)
{
	vector<T, N> out;
	for (int i = 0; i < N; ++i) {
		out[i] = v[i] / value;
	}
	return out;
}

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N>
div(const vector<T, N> &a, const vector<T, N> &b)
{
	vector<T, N> out;
	for (int i = 0; i < N; ++i) {
		out[i] = a[i] / b[i];
	}
	return out;
}

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N>
div_up(const vector<T, N> &a, const vector<T, N> &b)
{
	vector<T, N> out;
	for (int i = 0; i < N; ++i) {
		out[i] = (a[i] + b[i] - 1) / b[i];
	}
	return out;
}

//------------------------------------------------------------
// Product of all components
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline T
product(const vector<T, N> &v)
{
	T r = T{1};
	for (int i = 0; i < N; ++i) {
		r *= v[i];
	}
	return r;
}

template<typename T>
IANE_DECL_HYBRID inline typename std::enable_if<std::is_fundamental<T>::value, T>::type
product(const T &value)
{
	return value;
}

//------------------------------------------------------------
// Rounding helpers (round, floor, ceil)
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline vector<int, N>
round(const vector<T, N> &v)
{
	vector<int, N> out;
	for (int i = 0; i < N; ++i) {
#ifdef __CUDA_ARCH__
		out[i] = ::roundf(static_cast<float>(v[i]));
#else
		using std::round;
		out[i] = static_cast<int>(round(v[i]));
#endif
	}
	return out;
}

template<typename T, int N>
IANE_DECL_HYBRID inline vector<int, N>
floor(const vector<T, N> &v)
{
	vector<int, N> out;
	for (int i = 0; i < N; ++i) {
#ifdef __CUDA_ARCH__
		out[i] = ::floorf(static_cast<float>(v[i]));
#else
		using std::floor;
		out[i] = static_cast<int>(floor(v[i]));
#endif
	}
	return out;
}

template<typename T, int N>
IANE_DECL_HYBRID inline vector<int, N>
ceil(const vector<T, N> &v)
{
	vector<int, N> out;
	for (int i = 0; i < N; ++i) {
#ifdef __CUDA_ARCH__
		out[i] = ::ceilf(static_cast<float>(v[i]));
#else
		using std::ceil;
		out[i] = static_cast<int>(ceil(v[i]));
#endif
	}
	return out;
}

//------------------------------------------------------------
// Insert / remove dimension
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N + 1>
insert_dimension(const vector<T, N> &v, T value, int dim_index)
{
	IANE_ASSERT(dim_index >= 0);
	IANE_ASSERT(dim_index <= N);
	vector<T, N + 1> out;
	for (int i = 0; i < dim_index; ++i) {
		out[i] = v[i];
	}
	out[dim_index] = value;
	for (int i = dim_index; i < N; ++i) {
		out[i + 1] = v[i];
	}
	return out;
}

template<typename T, int N>
IANE_DECL_HYBRID inline vector<T, N - 1>
remove_dimension(const vector<T, N> &v, int dim_index)
{
	IANE_ASSERT(dim_index >= 0);
	IANE_ASSERT(dim_index < N);
	vector<T, N - 1> out;
	for (int i = 0; i < dim_index; ++i) {
		out[i] = v[i];
	}
	for (int i = dim_index; i < N - 1; ++i) {
		out[i] = v[i + 1];
	}
	return out;
}

//------------------------------------------------------------
// Sort components (simple bubble sort)
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID inline void
sort(vector<T, N> &v)
{
	for (int i = 0; i < N; ++i) {
		for (int j = 1; j < N - i; ++j) {
			if (v[j] < v[j - 1]) {
				using std::swap;
				swap(v[j], v[j - 1]);
			}
		}
	}
}

//------------------------------------------------------------
// Find an orthonormal vector to a 3D vector
//------------------------------------------------------------

template<MathVector V>
IANE_DECL_HYBRID inline auto
find_orthonormal(const V &v)
{
	static_assert(V::dim == 3, "find_orthonormal() requires 3D vector");
	using T = typename V::value_type;
	vector<T, 3> result{};
	for (int j = 0; j < 3; ++j) {
		if (v[j] != T{}) {
			auto x = v[j];
			auto y = v[(j + 1) % 3];
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

template<typename T>
IANE_DECL_HYBRID inline vector<T, 2>
rotate(const vector<T, 2> &v, double angle)
{
	using std::sin;
	using std::cos;
	auto sa = sin(angle);
	auto ca = cos(angle);

	return vector<T, 2>{
		static_cast<T>(ca * v[0] - sa * v[1]),
		static_cast<T>(sa * v[0] + ca * v[1])
	};
}

} // namespace iane::math
