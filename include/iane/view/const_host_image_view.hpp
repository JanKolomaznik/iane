#pragma once
#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/math/offset.hpp>

namespace iane {

template<typename T, int Dim>
class const_host_image_view
{
public:
	using value_type   = T;
	using extents_t    = typename math::dim_traits<Dim>::extents_t;
	using coord_t      = typename math::dim_traits<Dim>::coord_t;
	static constexpr int dimension = Dim;

	struct is_host_view_tag {};

	const_host_image_view() : m_ptr(nullptr), m_size{}, m_strides{} {}
	const_host_image_view(const T *ptr, extents_t size, extents_t strides)
		: m_ptr(ptr), m_size(size), m_strides(strides) {}

	extents_t dimensions() const { return m_size; }
	extents_t strides() const    { return m_strides; }
	const T *pointer() const     { return m_ptr; }

	const T &operator[](const coord_t &c) const
	{
		auto offset = math::offset_in_strided_memory(m_strides, c);
		return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_ptr) + offset);
	}

private:
	const T *m_ptr;
	extents_t m_size;
	extents_t m_strides;
};

} // namespace iane
