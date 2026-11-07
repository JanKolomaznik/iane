#pragma once
/**
 * @file vector.tcc
 * @brief Template definitions for iane::math::vector.
 */

namespace iane::math {

//------------------------------------------------------------
// Constructors
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr vector<TElement, tN>::vector()
{
	for (int i = 0; i < tN; ++i) {
		mElements[i] = TElement{};
	}
}

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr vector<TElement, tN>::vector(TElement aValue, std::true_type)
{
	for (int i = 0; i < tN; ++i) {
		mElements[i] = aValue;
	}
}

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr vector<TElement, tN>::vector(std::initializer_list<TElement> aInit)
{
	int i = 0;
	for (auto &v : aInit) {
		if (i < tN) {
			mElements[i++] = v;
		}
	}
	for (; i < tN; ++i) {
		mElements[i] = TElement{};
	}
}

template<typename TElement, int tN>
template<typename TOther>
IANE_DECL_HYBRID constexpr vector<TElement, tN>::vector(const vector<TOther, tN> &aOther)
{
	for (int i = 0; i < tN; ++i) {
		mElements[i] = static_cast<TElement>(aOther.mElements[i]);
	}
}

//------------------------------------------------------------
// Element access
//------------------------------------------------------------

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr TElement &vector<TElement, tN>::operator[](int aIdx)
{
	return mElements[aIdx];
}

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr const TElement &vector<TElement, tN>::operator[](int aIdx) const
{
	return mElements[aIdx];
}

template<typename TElement, int tN>
template<int tI>
IANE_DECL_HYBRID constexpr TElement &vector<TElement, tN>::get()
{
	static_assert(tI >= 0 && tI < tN, "vector: index out of range");
	return mElements[tI];
}

template<typename TElement, int tN>
template<int tI>
IANE_DECL_HYBRID constexpr const TElement &vector<TElement, tN>::get() const
{
	static_assert(tI >= 0 && tI < tN, "vector: index out of range");
	return mElements[tI];
}

//------------------------------------------------------------
// Arithmetic ops
//------------------------------------------------------------

template<typename TElement, int tN>
template<typename TOther>
IANE_DECL_HYBRID constexpr vector<TElement, tN> &
vector<TElement, tN>::operator+=(const vector<TOther, tN> &aRhs)
{
	for (int i = 0; i < tN; ++i) {
		mElements[i] += aRhs[i];
	}
	return *this;
}

template<typename TElement, int tN>
template<typename TOther>
IANE_DECL_HYBRID constexpr vector<TElement, tN> &
vector<TElement, tN>::operator-=(const vector<TOther, tN> &aRhs)
{
	for (int i = 0; i < tN; ++i) {
		mElements[i] -= aRhs[i];
	}
	return *this;
}

template<typename TElement, int tN>
IANE_DECL_HYBRID constexpr vector<TElement, tN>
vector<TElement, tN>::operator-() const
{
	vector<TElement, tN> out;
	for (int i = 0; i < tN; ++i) {
		out[i] = -mElements[i];
	}
	return out;
}

} // namespace iane::math
