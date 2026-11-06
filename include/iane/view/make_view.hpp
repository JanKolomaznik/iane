#pragma once
#include <iane/view/host_image_view.hpp>
#include <iane/view/const_host_image_view.hpp>
#include <iane/view/device_image_view.hpp>
#include <iane/view/const_device_image_view.hpp>
#include <iane/math/coord_traits.hpp>

namespace iane::host {

/* -------- HOST VIEWS -------- */

template<typename T, int Dim>
inline host_image_view<T, Dim>
make_view(T *ptr, typename math::dim_traits<Dim>::extents_t size)
{
	using ext = typename math::dim_traits<Dim>::extents_t;
	return {ptr, size, ext{}};
}

template<typename T, int Dim>
inline const_host_image_view<T, Dim>
make_const_view(const T *ptr, typename math::dim_traits<Dim>::extents_t size)
{
	using ext = typename math::dim_traits<Dim>::extents_t;
	return {ptr, size, ext{}};
}

} // namespace iane::host


namespace iane::device {

/* -------- DEVICE VIEWS -------- */

template<typename T, int Dim>
IANE_DECL_HOST device_image_view<T, Dim>
make_view(T *ptr, typename math::dim_traits<Dim>::extents_t size)
{
	using ext = typename math::dim_traits<Dim>::extents_t;
	return {ptr, size, ext{}};
}

template<typename T, int Dim>
IANE_DECL_HOST const_device_image_view<T, Dim>
make_const_view(const T *ptr, typename math::dim_traits<Dim>::extents_t size)
{
	using ext = typename math::dim_traits<Dim>::extents_t;
	return {ptr, size, ext{}};
}

} // namespace iane::device
