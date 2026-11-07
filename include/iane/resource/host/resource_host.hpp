#pragma once
/**
 * @file resource_host.hpp
 * @brief Host memory resource for iane::image.
 */

#include <memory>
#include <cstdint>

#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/math/vector_util.hpp>
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
	allocate(const extents_t &size)
	{
		using iane::math::coord_cast;
		using iane::math::product;

		const auto count = product(coord_cast<std::int64_t>(size));
		owner_t o;
		o.ptr.reset(new element_t[count]);
		o.size = size;
		return o;
	}

	// ---------------------------------------------------------------------
	// Pointer
	// ---------------------------------------------------------------------

	static IANE_DECL_HYBRID element_t *
	pointer(const owner_t &o)
	{
		return o.ptr.get();
	}

	// ---------------------------------------------------------------------
	// Strides (contiguous, element-major, converted to bytes)
	// ---------------------------------------------------------------------

	static IANE_DECL_HYBRID extents_t
	strides(const extents_t &size, const owner_t &)
	{
		extents_t s;

		if constexpr (tDim == 1) {
			s = 1;
		} else {
			s[0] = 1;
			for (int i = 1; i < tDim; ++i) {
				s[i] = s[i - 1] * size[i - 1];
			}
		}

		// Convert from element strides to byte strides
		for (int i = 0; i < tDim; ++i) {
			s[i] *= static_cast<int>(sizeof(element_t));
		}
		return s;
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
