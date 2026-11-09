#pragma once

#include <iane/algorithms/transform.hpp>
#include <iane/execution/prepared.hpp>

#include <limits>

namespace iane {

/// Boundary sampling for flat erosion and dilation.
struct morphology_options {
	border_mode border = border_mode::clamp;
};

/// A flat structuring element bounded by a per-axis radius.
template<class TCoord, class TPredicate>
struct flat_footprint {
	TCoord radius;
	TPredicate contains;
};

namespace detail {

// Select every offset inside a box.
struct include_offset {
	template<class TCoord>
	IANE_DECL_HYBRID bool operator()(const TCoord&) const { return true; }
};

template<class TSrc, class TDst, class TCoord, class TPredicate>
concept morphology_views = concepts::TransformViews<TSrc, TDst> &&
	std::same_as<TCoord, typename TSrc::coord_t> &&
	std::same_as<typename TSrc::value_type, typename TDst::value_type> &&
	std::is_arithmetic_v<typename TSrc::value_type> &&
	requires(const TPredicate& aContains, const TCoord& aOffset) {
		{ aContains(aOffset) } -> std::convertible_to<bool>;
	};

// Reduce selected samples to their minimum or maximum.
template<bool tDilate, class TCoord, class TPredicate>
struct morphology_filter {
	flat_footprint<TCoord, TPredicate> footprint;
	TCoord extent;

	IANE_CUDA_HD_WARNING_DISABLE
	template<concepts::Locator TLocator>
	IANE_DECL_HYBRID typename TLocator::value_type operator()(const TLocator& aLocator) const
	{
		using value_t = typename TLocator::value_type;
		using limits = std::numeric_limits<value_t>;
		constexpr auto lower = limits::has_infinity ? -limits::infinity() : limits::lowest();
		constexpr auto upper = limits::has_infinity ? limits::infinity() : limits::max();
		value_t result = tDilate ? lower : upper;
		TCoord position{};
		do {
			TCoord offset{};
			for (int axis = 0; axis < TLocator::dimension; ++axis) {
				math::detail::axis(offset, axis) =
					math::detail::axis(position, axis) - math::detail::axis(footprint.radius, axis);
			}
			if (footprint.contains(static_cast<const TCoord&>(offset))) {
				if constexpr (tDilate) {
					for (int axis = 0; axis < TLocator::dimension; ++axis) {
						math::detail::axis(offset, axis) = -math::detail::axis(offset, axis);
					}
				}
				const auto value = aLocator[offset];
				if (tDilate ? value > result : value < result) {
					result = value;
				}
			}
		} while (math::detail::advance<TLocator::dimension>(position, extent));
		return result;
	}
};

template<int tDim, class TCoord>
TCoord morphology_extent(TCoord aRadius)
{
	TCoord extent{};
	for (int axis = 0; axis < tDim; ++axis) {
		const auto radius = math::detail::axis(aRadius, axis);
		if (radius < 0 || radius > (std::numeric_limits<int>::max() - 1) / 2) {
			throw std::invalid_argument("Morphology requires a nonnegative radius with representable width");
		}
		math::detail::axis(extent, axis) = 2 * radius + 1;
	}
	return extent;
}

template<bool tDilate, class TSrc, class TDst, class TCoord, class TPredicate>
auto apply_morphology(TSrc aSource, TDst aDestination, flat_footprint<TCoord, TPredicate> aFootprint,
	morphology_options aOptions, view_policy<TDst> aPolicy)
{
	const auto extent = morphology_extent<TSrc::dimension>(aFootprint.radius);
	return transform_locator(aSource, aDestination,
		morphology_filter<tDilate, TCoord, TPredicate>{aFootprint, extent}, aOptions.border, aPolicy);
}

template<class TSrc, class TDst, class TCoord, class TPredicate, class TPolicy>
concept prepared_morphology_views = morphology_views<TSrc, TDst, TCoord, TPredicate> &&
	std::same_as<TCoord, typename math::dim_traits<TSrc::dimension>::coord_t> &&
	std::same_as<typename TDst::extents_t, typename math::dim_traits<TSrc::dimension>::extents_t> &&
	std::same_as<TPolicy, view_policy<TDst>>;

// Keep the shape and footprint for two passes sharing one intermediate image.
template<bool tDilateFirst, int tDim, class TValue, class TCoord, class TPredicate>
struct compound_morphology_state {
	typename math::dim_traits<tDim>::extents_t size;
	flat_footprint<TCoord, TPredicate> footprint;
	TCoord extent;
	border_mode border;
	std::size_t bytes;

	std::size_t scratch_bytes() const { return bytes; }

	template<class TSrc, class TDst, class TPolicy>
	requires prepared_morphology_views<TSrc, TDst, TCoord, TPredicate, TPolicy> &&
		(TSrc::dimension == tDim) && std::same_as<TValue, typename TSrc::value_type>
	auto run(TSrc aSource, TDst aDestination, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		const auto actual = matching_dimensions(aSource, aDestination);
		for (int axis = 0; axis < tDim; ++axis) {
			if (math::detail::axis(actual, axis) != math::detail::axis(size, axis)) {
				throw std::invalid_argument("Morphology views must match the prepared shape");
			}
		}
		auto intermediate = aScratch.template image<TValue, tDim>(size);
		transform_locator(aSource, intermediate,
			morphology_filter<tDilateFirst, TCoord, TPredicate>{footprint, extent}, border, aPolicy);
		return transform_locator(intermediate, aDestination,
			morphology_filter<!tDilateFirst, TCoord, TPredicate>{footprint, extent}, border, aPolicy);
	}
};

// Validate the prototype shape and reserve storage for the first pass.
template<bool tDilateFirst, class TCoord, class TPredicate>
struct compound_morphology_algorithm {
	flat_footprint<TCoord, TPredicate> footprint;
	morphology_options options;

	template<class TSrc, class TDst, class TPolicy>
	requires prepared_morphology_views<TSrc, TDst, TCoord, TPredicate, TPolicy>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy) const
	{
		using value_t = typename TSrc::value_type;
		constexpr int dimension = TSrc::dimension;
		const auto size = matching_dimensions(aSource, aDestination);
		const auto extent = morphology_extent<dimension>(footprint.radius);
		const auto count = resource::detail::packed_layout<value_t, dimension>(size).first;
		return compound_morphology_state<tDilateFirst, dimension, value_t, TCoord, TPredicate>{
			size, footprint, extent, options.border, count * sizeof(value_t)};
	}
};

} // namespace detail

/**
 * @brief Select a subset of the offsets within the given radius.
 * @param aRadius Nonnegative per-axis radius.
 * @param aContains Pure backend-callable predicate on an offset.
 * @pre The predicate selects at least one offset.
 */
template<class TCoord, class TPredicate>
auto make_flat_footprint(TCoord aRadius, TPredicate aContains)
{
	return flat_footprint<TCoord, TPredicate>{aRadius, aContains};
}

/// Select all offsets within the nonnegative per-axis radius.
template<class TCoord>
auto box_footprint(TCoord aRadius)
{
	return make_flat_footprint(aRadius, detail::include_offset{});
}

/**
 * @brief Write the minimum of samples at the selected offsets.
 * @param aSource Scalar readable view on the destination backend.
 * @param aDestination Disjoint writable view of matching shape and element type.
 * @param aFootprint Flat structuring element. Samples are taken at p + offset.
 * @param aOptions Logical view boundary rule.
 * @param aPolicy Scheduling or CUDA stream policy.
 * @return Completion handle. Keep storage alive through completion.
 * @throws std::invalid_argument For invalid radii or mismatched shapes.
 *
 * NaN samples are ignored. An all-NaN neighborhood returns positive infinity.
 */
template<class TSrc, class TDst, class TCoord, class TPredicate>
requires detail::morphology_views<TSrc, TDst, TCoord, TPredicate>
auto erode(TSrc aSource, TDst aDestination, flat_footprint<TCoord, TPredicate> aFootprint,
	morphology_options aOptions = {}, view_policy<TDst> aPolicy = {})
{
	return detail::apply_morphology<false>(aSource, aDestination, aFootprint, aOptions, aPolicy);
}

/**
 * @brief Write the maximum of samples at the reflected footprint offsets.
 *
 * Samples are taken at p - offset. The requirements of erode apply.
 * NaN samples are ignored. An all-NaN neighborhood returns negative infinity.
 */
template<class TSrc, class TDst, class TCoord, class TPredicate>
requires detail::morphology_views<TSrc, TDst, TCoord, TPredicate>
auto dilate(TSrc aSource, TDst aDestination, flat_footprint<TCoord, TPredicate> aFootprint,
	morphology_options aOptions = {}, view_policy<TDst> aPolicy = {})
{
	return detail::apply_morphology<true>(aSource, aDestination, aFootprint, aOptions, aPolicy);
}

/**
 * @brief Prepare flat opening, with erosion followed by dilation.
 * @param aFootprint Flat structuring element for both passes.
 * @param aOptions Border rule applied to both passes.
 * @return Algorithm for execution_context::prepare on same-rank, same-backend views.
 *
 * Reserves a packed scratch image of the input shape. Supports strided
 * views and exact in-place execution. Keep storage alive through completion.
 * Each run requires the prepared extents and the same scalar element type.
 */
template<class TCoord, class TPredicate>
auto opening(flat_footprint<TCoord, TPredicate> aFootprint, morphology_options aOptions = {})
{
	return detail::compound_morphology_algorithm<false, TCoord, TPredicate>{aFootprint, aOptions};
}

/// Prepare flat closing, with dilation followed by erosion. See opening for the view contract.
template<class TCoord, class TPredicate>
auto closing(flat_footprint<TCoord, TPredicate> aFootprint, morphology_options aOptions = {})
{
	return detail::compound_morphology_algorithm<true, TCoord, TPredicate>{aFootprint, aOptions};
}

} // namespace iane
