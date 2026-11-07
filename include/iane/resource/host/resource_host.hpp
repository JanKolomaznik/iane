#pragma once
/**
 * @file resource_host.hpp
 * @brief Host memory resource for iane::image.
 */

#include <memory>
#include <cstdint>

#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/resource/packed_layout.hpp>
#include <iane/view/host_image_view.hpp>
#include <iane/view/const_host_image_view.hpp>

namespace iane::resource {

/**
 * @brief Host memory resource for image<T, Dim, host_image_resource<T,Dim>>.
 */
template<typename TElement, int tDim>
struct host_image_resource {
	using element_t		= TElement;
	static constexpr int dim = tDim;

	using extents_t		= typename iane::math::dim_traits<tDim>::extents_t;
	using view_t		= host_image_view<TElement, tDim>;
	using const_view_t	= const_host_image_view<TElement, tDim>;

	struct owner_t {
		std::unique_ptr<element_t[]>	ptr;
		extents_t			size{};
	};

	// ---------------------------------------------------------------------
	// Allocation
	// ---------------------------------------------------------------------

	static owner_t
	allocate(const extents_t &aSize)
	{
		const auto count = detail::packed_layout<TElement, tDim>(aSize).first;
		owner_t o;
		o.ptr.reset(new element_t[count]);
		o.size = aSize;
		return o;
	}

	// ---------------------------------------------------------------------
	// Pointer
	// ---------------------------------------------------------------------

	static IANE_DECL_HOST element_t *
	pointer(const owner_t &aO)
	{
		return aO.ptr.get();
	}

	// ---------------------------------------------------------------------
	// Strides (contiguous, element-major, converted to bytes)
	// ---------------------------------------------------------------------

	static extents_t
	strides(const extents_t &aSize, const owner_t &)
	{
		return detail::packed_layout<TElement, tDim>(aSize).second;
	}
};

} // namespace iane::resource

namespace iane {

/**
 * @brief Convenient alias for host image.
 */
template<typename TElement, int tDim = 2>
using host_image = image<TElement, tDim, resource::host_image_resource<TElement, tDim>>;

} // namespace iane
