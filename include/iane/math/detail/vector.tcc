#pragma once
/**
 * @file vector.tcc
 * @brief Template definitions for iane::math::vector.
 */

namespace iane::math {

//------------------------------------------------------------
// Constructors
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID constexpr vector<T, N>::vector()
{
	for (int i = 0; i < N; ++i) {
		elems[i] = T{};
	}
}

template<typename T, int N>
IANE_DECL_HYBRID constexpr vector<T, N>::vector(T value, std::true_type)
{
	for (int i = 0; i < N; ++i) {
		elems[i] = value;
	}
}

template<typename T, int N>
IANE_DECL_HYBRID constexpr vector<T, N>::vector(std::initializer_list<T> init)
{
	int i = 0;
	for (auto &v : init) {
		if (i < N) {
			elems[i++] = v;
		}
	}
	for (; i < N; ++i) {
		elems[i] = T{};
	}
}

template<typename T, int N>
template<typename U>
IANE_DECL_HYBRID constexpr vector<T, N>::vector(const vector<U, N> &other)
{
	for (int i = 0; i < N; ++i) {
		elems[i] = static_cast<T>(other.elems[i]);
	}
}

//------------------------------------------------------------
// Element access
//------------------------------------------------------------

template<typename T, int N>
IANE_DECL_HYBRID constexpr T &vector<T, N>::operator[](int idx)
{
	return elems[idx];
}

template<typename T, int N>
IANE_DECL_HYBRID constexpr const T &vector<T, N>::operator[](int idx) const
{
	return elems[idx];
}

template<typename T, int N>
template<int I>
IANE_DECL_HYBRID constexpr T &vector<T, N>::get()
{
	static_assert(I >= 0 && I < N, "vector: index out of range");
	return elems[I];
}

template<typename T, int N>
template<int I>
IANE_DECL_HYBRID constexpr const T &vector<T, N>::get() const
{
	static_assert(I >= 0 && I < N, "vector: index out of range");
	return elems[I];
}

//------------------------------------------------------------
// Arithmetic ops
//------------------------------------------------------------

template<typename T, int N>
template<typename U>
IANE_DECL_HYBRID constexpr vector<T, N> &
vector<T, N>::operator+=(const vector<U, N> &rhs)
{
	for (int i = 0; i < N; ++i) {
		elems[i] += rhs[i];
	}
	return *this;
}

template<typename T, int N>
template<typename U>
IANE_DECL_HYBRID constexpr vector<T, N> &
vector<T, N>::operator-=(const vector<U, N> &rhs)
{
	for (int i = 0; i < N; ++i) {
		elems[i] -= rhs[i];
	}
	return *this;
}

template<typename T, int N>
IANE_DECL_HYBRID constexpr vector<T, N>
vector<T, N>::operator-() const
{
	vector<T, N> out;
	for (int i = 0; i < N; ++i) {
		out[i] = -elems[i];
	}
	return out;
}

} // namespace iane::math
