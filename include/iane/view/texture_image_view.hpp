#pragma once

#include <iane/math/coord_traits.hpp>

#if defined(__CUDACC__)
#include <concepts>

namespace iane {

/// Scalar element types supported by CUDA texture images.
template<class TElement>
concept texture_element = std::same_as<TElement, float> ||
	std::same_as<TElement, signed char> || std::same_as<TElement, unsigned char> ||
	std::same_as<TElement, short> || std::same_as<TElement, unsigned short> ||
	std::same_as<TElement, int> || std::same_as<TElement, unsigned int>;

/**
 * @brief Read-only 2D CUDA texture with point sampling and element-type reads.
 *
 * The owner must outlive all work using the view. Index in bounds on the
 * CUDA device associated with the texture object.
 */
template<texture_element TElement>
class texture_image_view {
public:
	using value_type = TElement;
	using coord_t = math::dim_traits<2>::coord_t;
	using extents_t = math::dim_traits<2>::extents_t;
	using is_device_view_tag = void;
	static constexpr int dimension = 2;

	/// Borrow a point-sampled texture with unnormalized coordinates.
	texture_image_view(cudaTextureObject_t aTexture, extents_t aSize)
		: mTexture(aTexture), mSize(aSize) {}

	/// Return the logical extent in texels.
	IANE_DECL_HYBRID extents_t dimensions() const { return mSize; }

	/// Read the texel at an in-bounds integer position.
	IANE_CUDA_DEVICE TElement operator[](const coord_t& aPosition) const
	{
		return tex2D<TElement>(mTexture, (mOrigin[0] + aPosition[0]) + 0.5f,
			(mOrigin[1] + aPosition[1]) + 0.5f);
	}

	/// Borrow an in-bounds region whose coordinates start at zero.
	texture_image_view subview(const coord_t& aCorner, const extents_t& aSize) const
	{
		texture_image_view result(mTexture, aSize);
		result.mOrigin = coord_t{mOrigin[0] + aCorner[0], mOrigin[1] + aCorner[1]};
		return result;
	}

private:
	cudaTextureObject_t mTexture;
	extents_t mSize;
	coord_t mOrigin{};
};

} // namespace iane
#endif
