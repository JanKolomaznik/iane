#pragma once
#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/math/offset.hpp>

namespace iane {

template<typename TElement, int tDim>
class const_device_image_view
{
public:
	using value_type   = TElement;
	using extents_t    = typename math::dim_traits<tDim>::extents_t;
	using coord_t      = typename math::dim_traits<tDim>::coord_t;
	static constexpr int dimension = tDim;

	struct is_device_view_tag {};

	IANE_DECL_HOST
	const_device_image_view() : mPtr(nullptr), mSize{}, mStrides{} {}

	IANE_DECL_HOST
	const_device_image_view(const TElement *aPtr, extents_t aSize, extents_t aStrides)
		: mPtr(aPtr), mSize(aSize), mStrides(aStrides) {}

	IANE_DECL_HYBRID
	const TElement &operator[](const coord_t &aC) const
	{
		auto offset = math::offset_in_strided_memory(mStrides, aC);
		return *reinterpret_cast<const TElement*>(reinterpret_cast<const char*>(mPtr) + offset);
	}

	IANE_DECL_HYBRID extents_t dimensions() const { return mSize; }
	IANE_DECL_HYBRID extents_t strides() const { return mStrides; }
	IANE_DECL_HYBRID const TElement *pointer() const { return mPtr; }

	/// Borrow a rectangular region. The caller must keep the region inside the view.
	IANE_DECL_HOST
	const_device_image_view subview(const coord_t& aCorner, const extents_t& aSize) const
	{
		const auto offset = math::offset_in_strided_memory(mStrides, aCorner);
		return {reinterpret_cast<const TElement*>(reinterpret_cast<const char*>(mPtr) + offset),
			aSize, mStrides};
	}

private:
	const TElement *mPtr;
	extents_t mSize;
	extents_t mStrides;
};

} // namespace iane
