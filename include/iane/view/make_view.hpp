#pragma once
#include <iane/view/host_image_view.hpp>
#include <iane/view/const_host_image_view.hpp>
#include <iane/view/device_image_view.hpp>
#include <iane/view/const_device_image_view.hpp>
#include <iane/resource/packed_layout.hpp>

namespace iane::host {

/* -------- HOST VIEWS -------- */

/**
 * @brief Borrow a packed buffer with the first axis contiguous.
 * @param aPtr Buffer holding the requested extent in the view's memory space.
 * @param aSize Nonnegative image extents.
 * @return View that remains valid while the buffer is alive.
 * @throws std::invalid_argument For negative extents.
 * @throws std::overflow_error If the packed layout is not representable.
 */
template<typename TElement, int tDim>
inline host_image_view<TElement, tDim>
make_view(TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	const auto layout = resource::detail::packed_layout<TElement, tDim>(aSize);
	return {aPtr, aSize, layout.second};
}

/// Borrow a packed buffer. See iane::host::make_view for layout and lifetime requirements.
template<typename TElement, int tDim>
inline const_host_image_view<TElement, tDim>
make_const_view(const TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	const auto layout = resource::detail::packed_layout<TElement, tDim>(aSize);
	return {aPtr, aSize, layout.second};
}

} // namespace iane::host


namespace iane::device {

/* -------- DEVICE VIEWS -------- */

/// Borrow a packed buffer. See iane::host::make_view for layout and lifetime requirements.
template<typename TElement, int tDim>
IANE_DECL_HOST device_image_view<TElement, tDim>
make_view(TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	const auto layout = resource::detail::packed_layout<TElement, tDim>(aSize);
	return {aPtr, aSize, layout.second};
}

/// Borrow a packed buffer. See iane::host::make_view for layout and lifetime requirements.
template<typename TElement, int tDim>
IANE_DECL_HOST const_device_image_view<TElement, tDim>
make_const_view(const TElement *aPtr, typename math::dim_traits<tDim>::extents_t aSize)
{
	const auto layout = resource::detail::packed_layout<TElement, tDim>(aSize);
	return {aPtr, aSize, layout.second};
}

} // namespace iane::device
