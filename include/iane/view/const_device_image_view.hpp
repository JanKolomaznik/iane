#pragma once
#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/math/offset.hpp>

namespace iane {

template<typename T, int Dim>
class const_device_image_view
{
public:
	using value_type   = T;
	using extents_t    = typename math::dim_traits<Dim>::extents_t;
	using coord_t      = typename math::dim_traits<Dim>::coord_t;

	struct is_device_view_tag {};

	IANE_DECL_HOST
	const_device_image_view() : m_ptr(nullptr), m_size{}, m_strides{} {}

	IANE_DECL_HOST
	const_device_image_view(const T *ptr, extents_t size, extents_t strides)
		: m_ptr(ptr), m_size(size), m_strides(strides) {}

	IANE_DECL_HYBRID
	const T &operator[](const coord_t &c) const
	{
		auto offset = math::offset_in_strided_memory(m_strides, c);
		return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_ptr) + offset);
	}

	IANE_DECL_HYBRID extents_t dimensions() const { return m_size; }
	IANE_DECL_HYBRID extents_t strides() const { return m_strides; }
	IANE_DECL_HYBRID const T *pointer() const { return m_ptr; }

private:
	const T *m_ptr;
	extents_t m_size;
	extents_t m_strides;
};

} // namespace iane
