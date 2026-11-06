#pragma once
/**
 * @file image.hpp
 * @brief Unified N-dimensional image with pluggable resource backend.
 *
 * Template parameters:
 *  - TElement : pixel / voxel element type
 *  - tDim     : dimensionality (1D, 2D, 3D, ...)
 *  - TResource: resource policy (host, device, Vulkan, ...)
 *
 * The resource type is responsible for:
 *  - ownership (owner_t)
 *  - pointer access
 *  - stride computation
 *  - view / const_view types
 */

#include <iane/config.hpp>
#include <iane/math/coord_traits.hpp>	// dim_traits
// Resource-specific headers are included by user code.

namespace iane {

/**
 * @brief Unified image with a resource policy.
 *
 * Resource type requirements (for TElement, tDim):
 *
 *  struct SomeResource {
 *  	using element_t   = TElement;
 *  	static constexpr int dim = tDim;
 *  	using extents_t   = iane::math::dim_traits<tDim>::extents_t;
 *  	using owner_t;
 *  	using view_t;
 *  	using const_view_t;
 *
 *  	static owner_t    allocate(extents_t size);
 *  	static element_t *pointer(owner_t &owner);
 *  	static extents_t  strides(const extents_t &size, const owner_t &owner);
 *  };
 */
template<typename TElement, int tDim, typename TResource>
class image {
public:
	using element_t		= TElement;
	using value_type	= TElement;
	static constexpr int dim = tDim;

	using resource_t	= TResource;
	using extents_t		= typename resource_t::extents_t;
	using owner_t		= typename resource_t::owner_t;
	using view_t		= typename resource_t::view_t;
	using const_view_t	= typename resource_t::const_view_t;

public:
	IANE_DECL_HOST image() = default;

	/// Construct from extents
	IANE_DECL_HOST explicit image(extents_t size)
		: mSize(size)
		, mOwner(resource_t::allocate(size))
		, mStrides(resource_t::strides(size, mOwner))
	{}

	/// Convenience: 1D image
	IANE_DECL_HOST explicit image(int s0)
		: image(extents_t{s0})
	{
		static_assert(tDim == 1, "Image dimensionality mismatch for 1D ctor");
	}

	/// Convenience: 2D image
	IANE_DECL_HOST image(int s0, int s1)
		: image(extents_t{s0, s1})
	{
		static_assert(tDim == 2, "Image dimensionality mismatch for 2D ctor");
	}

	/// Convenience: 3D image
	IANE_DECL_HOST image(int s0, int s1, int s2)
		: image(extents_t{s0, s1, s2})
	{
		static_assert(tDim == 3, "Image dimensionality mismatch for 3D ctor");
	}

	IANE_DECL_HOST image(const image &) = delete;
	IANE_DECL_HOST image &operator=(const image &) = delete;

	IANE_DECL_HOST image(image &&) = default;
	IANE_DECL_HOST image &operator=(image &&) = default;

	// ---------------------------------------------------------------------
	// Introspection
	// ---------------------------------------------------------------------

	IANE_DECL_HYBRID extents_t
	dimensions() const
	{
		return mSize;
	}

	IANE_DECL_HYBRID extents_t
	strides() const
	{
		return mStrides;
	}

	IANE_DECL_HYBRID value_type *
	pointer() const
	{
		return resource_t::pointer(mOwner);
	}

	// ---------------------------------------------------------------------
	// View access
	// ---------------------------------------------------------------------

	IANE_DECL_HYBRID view_t
	view() const
	{
		return view_t(pointer(), dimensions(), strides());
	}

	IANE_DECL_HYBRID const_view_t
	const_view() const
	{
		return const_view_t(pointer(), dimensions(), strides());
	}

	// ---------------------------------------------------------------------
	// Resize / reallocate
	// ---------------------------------------------------------------------

	IANE_DECL_HOST void
	resize(extents_t new_size)
	{
		mSize    = new_size;
		mOwner   = resource_t::allocate(new_size);
		mStrides = resource_t::strides(new_size, mOwner);
	}

	IANE_DECL_HOST void
	reset()
	{
		mOwner   = owner_t{};
		mSize    = extents_t{};
		mStrides = extents_t{};
	}

private:
	extents_t	mSize{};
	owner_t		mOwner{};
	extents_t	mStrides{};
};

// -------------------------------------------------------------------------
// Free functions view() and const_view() – generic image access
// -------------------------------------------------------------------------

template<typename TImage>
inline IANE_DECL_HYBRID auto
view(const TImage &img) -> typename TImage::view_t
{
	return img.view();
}

template<typename TImage>
inline IANE_DECL_HYBRID auto
const_view(const TImage &img) -> typename TImage::const_view_t
{
	return img.const_view();
}

} // namespace iane
