#pragma once

#include <iane/math/coord_traits.hpp>
#include <iane/math/vector_ops.hpp>

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
 * @brief Read-only 1D, 2D, or 3D CUDA texture with point sampling and element-type reads.
 *
 * The owner must outlive all work using the view. Index in bounds on the
 * CUDA device associated with the texture object.
 */
template<texture_element TElement, int tDim = 2>
requires (tDim >= 1 && tDim <= 3)
class texture_image_view {
public:
	using value_type = TElement;
	using coord_t = typename math::dim_traits<tDim>::coord_t;
	using extents_t = typename math::dim_traits<tDim>::extents_t;
	using is_device_view_tag = void;
	static constexpr int dimension = tDim;

	/// Borrow a point-sampled texture with unnormalized coordinates.
	texture_image_view(cudaTextureObject_t aTexture, extents_t aSize)
		: mTexture(aTexture), mSize(aSize) {}

	/// Return the logical extent in texels.
	IANE_DECL_HYBRID extents_t dimensions() const { return mSize; }

	/// Read the texel at an in-bounds integer position.
	IANE_CUDA_DEVICE TElement operator[](const coord_t& aPosition) const
	{
		if constexpr (tDim == 1) {
			return tex1D<TElement>(mTexture, (mOrigin + aPosition) + 0.5f);
		} else if constexpr (tDim == 2) {
			return tex2D<TElement>(mTexture, (mOrigin[0] + aPosition[0]) + 0.5f,
				(mOrigin[1] + aPosition[1]) + 0.5f);
		} else {
			return tex3D<TElement>(mTexture, (mOrigin[0] + aPosition[0]) + 0.5f,
				(mOrigin[1] + aPosition[1]) + 0.5f, (mOrigin[2] + aPosition[2]) + 0.5f);
		}
	}

	/// Borrow an in-bounds region whose coordinates start at zero.
	texture_image_view subview(const coord_t& aCorner, const extents_t& aSize) const
	{
		texture_image_view result(mTexture, aSize);
		result.mOrigin = mOrigin + aCorner;
		return result;
	}

private:
	cudaTextureObject_t mTexture;
	extents_t mSize;
	coord_t mOrigin{};
};

} // namespace iane
#endif
