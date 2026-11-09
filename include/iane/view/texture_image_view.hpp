#pragma once

#include <iane/math/coord_traits.hpp>
#include <iane/math/index.hpp>
#include <iane/math/vector_ops.hpp>

#if defined(__CUDACC__)
#include <cmath>
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
	using sample_coord_t = typename math::dim_traits<tDim>::float_coord_t;
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
		sample_coord_t position{};
		for (int i = 0; i < tDim; ++i) {
			math::detail::axis(position, i) =
				math::detail::axis(mOrigin, i) + math::detail::axis(aPosition, i) + 0.5f;
		}
		return fetch(position);
	}

	/**
	 * @brief Point sample, clamped to the edge texel centers of this view.
	 * @param aPosition View-local coordinate measured from the outer corner.
	 * @return Sampled texel, or zero for an empty view or nonfinite coordinate.
	 */
	IANE_CUDA_DEVICE TElement sample(sample_coord_t aPosition) const
	{
		using math::detail::axis;
		for (int i = 0; i < tDim; ++i) {
			auto& x = axis(aPosition, i);
			const auto extent = axis(mSize, i);
			if (extent <= 0 || !std::isfinite(x)) {
				return {};
			}
			const float last = extent - 0.5f;
			x = (x < 0.5f ? 0.5f : (x > last ? last : x)) + axis(mOrigin, i);
		}
		return fetch(aPosition);
	}

	/// Borrow an in-bounds region whose coordinates start at zero.
	texture_image_view subview(const coord_t& aCorner, const extents_t& aSize) const
	{
		texture_image_view result(mTexture, aSize);
		result.mOrigin = mOrigin + aCorner;
		return result;
	}

private:
	IANE_CUDA_DEVICE TElement fetch(const sample_coord_t& aPosition) const
	{
		if constexpr (tDim == 1) {
			return tex1D<TElement>(mTexture, aPosition);
		} else if constexpr (tDim == 2) {
			return tex2D<TElement>(mTexture, aPosition[0], aPosition[1]);
		} else {
			return tex3D<TElement>(mTexture, aPosition[0], aPosition[1], aPosition[2]);
		}
	}

	cudaTextureObject_t mTexture;
	extents_t mSize;
	coord_t mOrigin{};
};

} // namespace iane
#endif
