#pragma once
/**
 * @file resource_cuda.hpp
 * @brief CUDA device memory resource for iane::image.
 *
 * This header is intended to be used in CUDA-enabled builds.
 * It adapts the legacy memory_management<T,Dim>::device_memory_owner
 * interface into the new resource-based image design.
 */

#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/image/device_image_view.hpp>	// you will port/define this
#include <iane/memory/memory.hpp>		// you will port: memory_management<T,Dim>

namespace iane::resource {

/**
 * @brief CUDA device memory resource for image<T,Dim,device_image_resource<T,Dim>>.
 *
 * This resource assumes an existing memory_management<T,Dim> with a nested
 * device_memory_owner type, compatible with the old CUGIP implementation:
 *
 *  memory_management<T,Dim>::device_memory_owner owner;
 *  - construct from extents_t
 *  - extents_t dimensions() const;
 *  - extents_t strides() const;
 *  - some pointer-like member (e.g. mData.get()) for raw device pointer.
 */
template<typename TElement, int tDim>
struct device_image_resource {
	using element_t		= TElement;
	static constexpr int dim = tDim;

	using extents_t		= typename iane::math::dim_traits<tDim>::extents_t;
	using view_t		= device_image_view<TElement, tDim>;
	using const_view_t	= const_device_image_view<TElement, tDim>;

	using owner_t		= typename memory_management<TElement, tDim>::device_memory_owner;

	// ---------------------------------------------------------------------
	// Allocation
	// ---------------------------------------------------------------------

	static owner_t
	allocate(const extents_t &size)
	{
		// Assumes device_memory_owner(extents_t) ctor exists (as in original CUGIP)
		return owner_t(size);
	}

	// ---------------------------------------------------------------------
	// Pointer
	// ---------------------------------------------------------------------

	static IANE_DECL_HYBRID element_t *
	pointer(owner_t &o)
	{
		// Adaptation point: we assume an mData member with get().
		// When you port memory.hpp into IANE, you can refine this.
		return o.mData.get();
	}

	// ---------------------------------------------------------------------
	// Strides (query from owner)
	// ---------------------------------------------------------------------

	static IANE_DECL_HYBRID extents_t
	strides(const extents_t &, const owner_t &o)
	{
		// Assumes device_memory_owner::strides() exists (as in original CUGIP)
		return o.strides();
	}
};

} // namespace iane::resource

namespace iane {

/**
 * @brief Convenient alias for device image.
 */
template<typename TElement, int tDim = 2>
using device_image = image<TElement, tDim, resource::device_image_resource<TElement, tDim>>;

} // namespace iane
