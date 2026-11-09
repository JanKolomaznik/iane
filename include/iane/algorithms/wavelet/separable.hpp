#pragma once

#include <iane/algorithms/copy.hpp>
#include <iane/execution/prepared.hpp>
#include <iane/view/procedural.hpp>

namespace iane::detail {

template<class TSrc, class TDst, class TPolicy>
concept separable_wavelet_views = concepts::TransformViews<TSrc, TDst> &&
	TSrc::dimension == 2 && std::same_as<typename TSrc::coord_t, math::vec2i> &&
	std::same_as<typename TSrc::value_type, float> && std::same_as<typename TDst::value_type, float> &&
	concepts::WritableView<TDst, float> && std::same_as<TPolicy, view_policy<TDst>>;

// Keep the planned shape and reserve one full-size horizontal intermediate.
template<template<int, class> class TPass>
struct separable_wavelet_state {
	math::vec2i size;
	std::size_t bytes;

	std::size_t scratch_bytes() const { return bytes; }

	template<class TSrc, class TDst, class TPolicy>
	requires separable_wavelet_views<TSrc, TDst, TPolicy>
	auto run(TSrc aSource, TDst aDestination, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		const auto extent = matching_dimensions(aSource, aDestination);
		if (extent[0] != size[0] || extent[1] != size[1]) {
			throw std::invalid_argument("Wavelet views must match the prepared shape");
		}
		auto intermediate = aScratch.template image<float, 2>(size);
		copy(generate_view(intermediate, TPass<0, TSrc>{aSource, size[0] / 2}),
			intermediate, aPolicy);
		return copy(generate_view(aDestination, TPass<1, decltype(intermediate)>{
			intermediate, size[1] / 2}), aDestination, aPolicy);
	}
};

// Validate and size the reusable single-level separable wavelet.
template<template<int, class> class TPass, bool tAnalysis = true>
struct separable_wavelet_algorithm {
	template<class TSrc, class TDst, class TPolicy>
	requires separable_wavelet_views<TSrc, TDst, TPolicy>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy) const
	{
		const auto size = matching_dimensions(aSource, aDestination);
		if (size[0] <= 0 || size[1] <= 0 || size[0] % 2 != 0 || size[1] % 2 != 0) {
			throw std::invalid_argument("Wavelet transform requires positive even dimensions");
		}
		return separable_wavelet_state<TPass>{size, resource::detail::packed_layout<float, 2>(size).first * sizeof(float)};
	}
};

} // namespace iane::detail
