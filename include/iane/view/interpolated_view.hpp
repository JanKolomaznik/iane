#pragma once

#include <iane/locator/image_locator.hpp>
#include <iane/math/coord_traits.hpp>
#include <iane/view/procedural.hpp>

#include <cmath>

namespace iane {

/// Select the nearest texel, resolving ties toward the higher index.
struct nearest_t {};
/// Nearest-neighbor interpolation tag.
inline constexpr nearest_t nearest{};

/// Interpolate scalar texels linearly along every axis.
struct linear_t {};
/// Multilinear interpolation tag.
inline constexpr linear_t linear{};

namespace detail {

// Add continuous sampling while preserving the borrowed view's discrete reads.
template<class TView, class TMethod>
struct interpolated_view : procedural_backend<concepts::HostView<TView>> {
	using coord_t = typename TView::coord_t;
	using extents_t = typename TView::extents_t;
	using value_type = typename TView::value_type;
	using sample_coord_t = typename math::dim_traits<TView::dimension>::float_coord_t;
	using sample_value_type = std::conditional_t<std::same_as<TMethod, linear_t> &&
		std::is_integral_v<value_type>, float, value_type>;
	static constexpr int dimension = TView::dimension;

	TView source;
	border_mode border;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID extents_t dimensions() const { return source.dimensions(); }

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID auto operator[](const coord_t& aPosition) const
		-> decltype(source[aPosition])
	{
		return source[aPosition];
	}

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID sample_value_type sample(const sample_coord_t& aPosition) const
	{
		using math::detail::axis;
		const auto extent = source.dimensions();
		coord_t base{};
		sample_coord_t weights{};
		for (int i = 0; i < dimension; ++i) {
			auto position = static_cast<double>(axis(aPosition, i));
			const auto size = axis(extent, i);
			if (size <= 0 || !std::isfinite(position)) {
				return {};
			}
			if constexpr (std::same_as<TMethod, linear_t>) {
				position -= 0.5;
			}
			if (border == border_mode::clamp) {
				position = position < 0 ? 0 : (position > size - 1 ? size - 1 : position);
			}
			constexpr int first = std::same_as<TMethod, linear_t> ? -1 : 0;
			if (position < first || position >= size) {
				return {};
			}
			const auto lower = std::floor(position);
			axis(base, i) = static_cast<int>(lower);
			axis(weights, i) = static_cast<float>(position - lower);
		}
		const image_locator<TView> locator(source, coord_t{}, border);
		if constexpr (std::same_as<TMethod, nearest_t>) {
			return locator[base];
		} else {
			return interpolate<0>(locator, weights, base);
		}
	}

	// Combine the two taps along each axis. Skip taps with zero weight.
	template<int tAxis>
	IANE_DECL_HYBRID static sample_value_type interpolate(const image_locator<TView>& aLocator,
		const sample_coord_t& aWeights, coord_t aOffset)
	{
		if constexpr (tAxis == dimension) {
			return static_cast<sample_value_type>(aLocator[aOffset]);
		} else {
			const auto weight = static_cast<sample_value_type>(math::detail::axis(aWeights, tAxis));
			const auto lower = interpolate<tAxis + 1>(aLocator, aWeights, aOffset);
			if (weight == 0) {
				return lower;
			}
			++math::detail::axis(aOffset, tAxis);
			const auto upper = interpolate<tAxis + 1>(aLocator, aWeights, aOffset);
			return (1 - weight) * lower + weight * upper;
		}
	}
};

} // namespace detail

/**
 * @brief Borrow a readable view with software sampling at continuous positions.
 * @tparam TMethod Nearest or linear interpolation tag.
 * @param aSource Input view with nonnegative extents.
 * @param aBorder Border rule applied to individual taps at the logical boundary.
 * @return An interpolated view with the source backend and integer indexing.
 * @throws std::invalid_argument If an extent is negative.
 * @throws std::overflow_error If the extent product overflows.
 *
 * The outer corner is zero and texel centers are at half integers. Nearest
 * preserves the value type. Linear accepts arithmetic texels and promotes
 * integral texels to float. Empty views and nonfinite coordinates sample
 * as zero. Keep the borrowed source alive through evaluation and completion.
 */
template<concepts::ReadableView TView, class TMethod>
requires detail::procedural_domain<TView> && std::default_initializable<typename TView::value_type> &&
	(std::same_as<TMethod, nearest_t> ||
		(std::same_as<TMethod, linear_t> && std::is_arithmetic_v<typename TView::value_type>))
auto make_interpolated_view(TView aSource, TMethod, border_mode aBorder = border_mode::clamp)
{
	math::detail::element_count<TView::dimension>(aSource.dimensions());
	return detail::interpolated_view<TView, TMethod>{{}, aSource, aBorder};
}

} // namespace iane
