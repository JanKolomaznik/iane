#pragma once
#include <iane/view/host_image_view.hpp>
#include <iane/view/const_host_image_view.hpp>
#include <iane/view/device_image_view.hpp>
#include <iane/view/const_device_image_view.hpp>
#include <iane/math/coord_traits.hpp>

namespace iane::host {

/* -------- HOST VIEWS -------- */

template<typename TElement, int tDim>
inline host_image_view<TElement, tDim>
make_view(TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	using ext = typename math::dim_traits<tDim>::extents_t;
	return {aPtr, aSize, ext{}};
}

template<typename TElement, int tDim>
inline const_host_image_view<TElement, tDim>
make_const_view(const TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	using ext = typename math::dim_traits<tDim>::extents_t;
	return {aPtr, aSize, ext{}};
}

} // namespace iane::host


namespace iane::device {

/* -------- DEVICE VIEWS -------- */

template<typename TElement, int tDim>
IANE_DECL_HOST device_image_view<TElement, tDim>
make_view(TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	using ext = typename math::dim_traits<tDim>::extents_t;
	return {aPtr, aSize, ext{}};
}

template<typename TElement, int tDim>
IANE_DECL_HOST const_device_image_view<TElement, tDim>
make_const_view(const TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	using ext = typename math::dim_traits<tDim>::extents_t;
	return {aPtr, aSize, ext{}};
}

} // namespace iane::device
