#pragma once
#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/math/offset.hpp>

namespace iane {

/**
 * @brief Host image view – CPU only, no device qualifiers.
 */
template<typename T, int Dim>
class host_image_view
{
public:
	using value_type   = T;
	using extents_t    = typename math::dim_traits<Dim>::extents_t;
	using coord_t      = typename math::dim_traits<Dim>::coord_t;
	using diff_t       = typename math::dim_traits<Dim>::diff_t;
	static constexpr int dimension = Dim;

	struct is_host_view_tag {};

	host_image_view() : m_ptr(nullptr), m_size{}, m_strides{} {}
	host_image_view(T *ptr, extents_t size, extents_t strides)
		: m_ptr(ptr), m_size(size), m_strides(strides) {}

	extents_t dimensions() const { return m_size; }
	extents_t strides() const    { return m_strides; }
	T *pointer() const           { return m_ptr; }

	/**
	 * @brief Host-only element access.
	 */
	T &operator[](const coord_t &c) const
	{
		auto offset = math::offset_in_strided_memory(m_strides, c);
		return *reinterpret_cast<T*>(reinterpret_cast<char*>(m_ptr) + offset);
	}

	/**
	 * @brief Create subview using corner + size.
	 */
	host_image_view subview(const coord_t &corner, const extents_t &size) const
	{
		auto offset = math::offset_in_strided_memory(m_strides, corner);
		return {
			reinterpret_cast<T*>(reinterpret_cast<char*>(m_ptr) + offset),
			size,
			m_strides
		};
	}

	/**
	 * @brief Slice along dimension I.
	 */
	template<int I>
	host_image_view<T, Dim-1> slice(int index) const
	{
		static_assert(I >= 0 && I < Dim);
		typename math::dim_traits<Dim>::coord_t c{};
		c[I] = index;

		auto offset = math::offset_in_strided_memory(m_strides, c);

		return {
			reinterpret_cast<T*>(reinterpret_cast<char*>(m_ptr) + offset),
			remove_dimension(m_size, I),
			remove_dimension(m_strides, I)
		};
	}

private:
	T *m_ptr;
	extents_t m_size;
	extents_t m_strides;
};

} // namespace iane
