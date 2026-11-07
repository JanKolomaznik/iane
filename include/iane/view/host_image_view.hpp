#pragma once
#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/math/offset.hpp>

namespace iane {

/**
 * @brief Host image view – CPU only, no device qualifiers.
 */
template<typename TElement, int tDim>
class host_image_view
{
public:
	using value_type   = TElement;
	using extents_t    = typename math::dim_traits<tDim>::extents_t;
	using coord_t      = typename math::dim_traits<tDim>::coord_t;
	using diff_t       = typename math::dim_traits<tDim>::diff_t;
	static constexpr int dimension = tDim;

	struct is_host_view_tag {};

	host_image_view() : mPtr(nullptr), mSize{}, mStrides{} {}
	host_image_view(TElement *aPtr, extents_t aSize, extents_t aStrides)
		: mPtr(aPtr), mSize(aSize), mStrides(aStrides) {}

	extents_t dimensions() const { return mSize; }
	extents_t strides() const    { return mStrides; }
	TElement *pointer() const           { return mPtr; }

	/**
	 * @brief Host-only element access.
	 */
	TElement &operator[](const coord_t &aC) const
	{
		auto offset = math::offset_in_strided_memory(mStrides, aC);
		return *reinterpret_cast<TElement*>(reinterpret_cast<char*>(mPtr) + offset);
	}

	/**
	 * @brief Create subview using corner + size.
	 */
	host_image_view subview(const coord_t &aCorner, const extents_t &aSize) const
	{
		auto offset = math::offset_in_strided_memory(mStrides, aCorner);
		return {
			reinterpret_cast<TElement*>(reinterpret_cast<char*>(mPtr) + offset),
			aSize,
			mStrides
		};
	}

	/**
	 * @brief Fix one axis and borrow the remaining dimensions in their original order.
	 * @tparam tI Axis to remove.
	 * @param aIndex In-bounds coordinate on that axis.
	 * @return View sharing the original storage and remaining byte strides.
	 */
	template<int tI>
	requires (tDim > 1 && tI >= 0 && tI < tDim)
	host_image_view<TElement, tDim - 1> slice(int aIndex) const
	{
		using slice_t = host_image_view<TElement, tDim - 1>;
		typename slice_t::extents_t size{};
		typename slice_t::extents_t strides{};
		for (int i = 0, j = 0; i < tDim; ++i) {
			if (i != tI) {
				math::detail::axis(size, j) = mSize[i];
				math::detail::axis(strides, j) = mStrides[i];
				++j;
			}
		}
		const auto offset = static_cast<std::ptrdiff_t>(mStrides[tI]) * aIndex;
		return {reinterpret_cast<TElement*>(reinterpret_cast<char*>(mPtr) + offset), size, strides};
	}

private:
	TElement *mPtr;
	extents_t mSize;
	extents_t mStrides;
};

} // namespace iane
