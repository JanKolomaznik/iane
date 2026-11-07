#pragma once

#include <iane/algorithms/copy.hpp>
#include <iane/execution/prepared.hpp>
#include <iane/view/procedural.hpp>

namespace iane {
namespace detail {

// Calculate a low- or high-frequency coefficient along one axis.
template<int tAxis, class TSource>
struct haar_pass {
	TSource source;
	int half;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID float operator()(math::vec2i aPosition) const
	{
		const bool high = aPosition[tAxis] >= half;
		aPosition[tAxis] = 2 * (aPosition[tAxis] - (high ? half : 0));
		const float first = source[aPosition];
		++aPosition[tAxis];
		const float second = source[aPosition];
		return (high ? first - second : first + second) * 0.7071067811865475f;
	}
};

template<class TSrc, class TDst, class TPolicy>
concept haar_views = concepts::TransformViews<TSrc, TDst> &&
	TSrc::dimension == 2 && std::same_as<typename TSrc::coord_t, math::vec2i> &&
	std::same_as<typename TSrc::value_type, float> && std::same_as<typename TDst::value_type, float> &&
	concepts::WritableView<TDst, float> && std::same_as<TPolicy, view_policy<TDst>>;

// Keep the planned shape and reserve one full-size horizontal intermediate.
struct haar_state {
	math::vec2i size;
	std::size_t bytes;

	std::size_t scratch_bytes() const { return bytes; }

	template<class TSrc, class TDst, class TPolicy>
	requires haar_views<TSrc, TDst, TPolicy>
	auto run(TSrc aSource, TDst aDestination, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		const auto extent = matching_dimensions(aSource, aDestination);
		if (extent[0] != size[0] || extent[1] != size[1]) {
			throw std::invalid_argument("Haar views must match the prepared shape");
		}
		auto intermediate = aScratch.template image<float, 2>(size);
		copy(generate_view(intermediate, haar_pass<0, TSrc>{aSource, size[0] / 2}),
			intermediate, aPolicy);
		return copy(generate_view(aDestination, haar_pass<1, decltype(intermediate)>{
			intermediate, size[1] / 2}), aDestination, aPolicy);
	}
};

// Validate and size the reusable single-level Haar operation.
struct haar_algorithm {
	template<class TSrc, class TDst, class TPolicy>
	requires haar_views<TSrc, TDst, TPolicy>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy) const
	{
		const auto size = matching_dimensions(aSource, aDestination);
		if (size[0] <= 0 || size[1] <= 0 || size[0] % 2 != 0 || size[1] % 2 != 0) {
			throw std::invalid_argument("Haar transform requires positive even dimensions");
		}
		return haar_state{size, resource::detail::packed_layout<float, 2>(size).first * sizeof(float)};
	}
};

} // namespace detail

/// Prepare a single-level orthonormal 2D Haar transform, including in-place use.
inline constexpr detail::haar_algorithm wavelet_forward{};

} // namespace iane
