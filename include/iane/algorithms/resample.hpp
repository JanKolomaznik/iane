#pragma once

#include <iane/algorithms/for_each.hpp>
#include <iane/math/coord_traits.hpp>

namespace iane {
namespace detail {

// Map each destination center to the source before assigning its sample.
template<class TSource, class TDestination, class TMap>
struct resample_pixel {
	TSource source;
	TMap map;

	IANE_CUDA_HD_WARNING_DISABLE
	template<class TValue>
	IANE_DECL_HYBRID void operator()(TValue&& aOutput, const typename TDestination::coord_t& aIndex) const
	{
		typename math::dim_traits<TDestination::dimension>::float_coord_t center{};
		for (int i = 0; i < TDestination::dimension; ++i) {
			math::detail::axis(center, i) = math::detail::axis(aIndex, i) + 0.5f;
		}
		const auto& position = center;
		static_cast<TValue&&>(aOutput) = source.sample(map(position));
	}
};

} // namespace detail

/**
 * @brief Map destination pixel centers to continuous source positions and sample them.
 * @param aSource Interpolated view on the destination backend.
 * @param aDestination Writable view with distinct output locations.
 * @param aMap Pure backend-callable map from destination centers to source positions.
 * @param aPolicy Backend scheduling or stream policy.
 * @return Completion handle for the destination writes.
 *
 * The map receives the float coordinate type of the destination rank and
 * both coordinate spaces use the outer corner as zero. Ranks may differ.
 * Filtering and boundaries follow the source. Keep all borrowed storage
 * alive through completion. Source and destination storage must be disjoint.
 */
template<concepts::InterpolatedView TSource, concepts::ViewDomain TDst, class TMap>
requires concepts::SameViewBackend<TSource, TDst> &&
	requires(const TSource& aSource, const TDst& aDst, const TMap& aMap,
		const typename math::dim_traits<TDst::dimension>::float_coord_t& aCenter,
		const typename TDst::coord_t& aIndex) {
		aDst[aIndex] = aSource.sample(aMap(aCenter));
	}
auto resample(TSource aSource, TDst aDestination, TMap aMap, view_policy<TDst> aPolicy = {})
{
	return for_each_position(aDestination, detail::resample_pixel<TSource, TDst, TMap>{aSource, aMap}, aPolicy);
}

} // namespace iane
