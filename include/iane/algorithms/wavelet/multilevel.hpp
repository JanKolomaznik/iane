#pragma once

#include <iane/algorithms/wavelet/separable.hpp>
#include <iane/image/pyramid.hpp>

#include <array>

namespace iane {
namespace detail {

inline math::vec2i wavelet_half(math::vec2i aSize)
{
	return math::vec2i{aSize[0] / 2, aSize[1] / 2};
}

inline void check_wavelet_levels(math::vec2i aSize, std::size_t aLevels)
{
	if (aLevels == 0) {
		throw std::invalid_argument("Multilevel wavelet requires at least one level");
	}
	for (std::size_t i = 0; i < aLevels; ++i) {
		if (aSize[0] <= 0 || aSize[1] <= 0 || aSize[0] % 2 != 0 || aSize[1] % 2 != 0) {
			throw std::invalid_argument("Wavelet extents must be positive and even at each level");
		}
		aSize = wavelet_half(aSize);
	}
}

template<class TView>
concept wavelet_subview = concepts::ViewDomain<TView> &&
	std::same_as<typename TView::extents_t, math::vec2i> &&
	requires(const TView& aView, math::vec2i aCorner, math::vec2i aSize) {
		{ aView.subview(aCorner, aSize) } -> std::same_as<TView>;
	};

// Reuse the same scratch region while traversing nested LL views.
template<class TState, bool tAnalysis>
struct multilevel_wavelet_state {
	math::vec2i size;
	std::vector<TState> levels;
	std::size_t bytes;

	std::size_t scratch_bytes() const { return bytes; }

	template<class TSrc, wavelet_subview TDst, class TPolicy>
	requires separable_wavelet_views<TSrc, TDst, TPolicy>
	auto run(TSrc aSource, TDst aDestination, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		const auto extent = matching_dimensions(aSource, aDestination);
		if (extent[0] != size[0] || extent[1] != size[1]) {
			throw std::invalid_argument("Wavelet views must match the prepared shape");
		}

		if constexpr (tAnalysis) {
			auto first_workspace = aScratch;
			auto result = levels.front().run(aSource, aDestination, first_workspace, aPolicy);
			for (std::size_t i = 1; i < levels.size(); ++i) {
				auto ll = aDestination.subview(math::vec2i{}, levels[i].size);
				auto workspace = aScratch;
				result = levels[i].run(ll, ll, workspace, aPolicy);
			}
			return result;
		} else {
			if (levels.size() == 1) {
				return levels.front().run(aSource, aDestination, aScratch, aPolicy);
			}
			if (!same_layout(aSource, aDestination)) {
				copy(aSource, aDestination, aPolicy);
			}
			for (std::size_t i = levels.size() - 1; i > 0; --i) {
				auto ll = aDestination.subview(math::vec2i{}, levels[i].size);
				auto workspace = aScratch;
				levels[i].run(ll, ll, workspace, aPolicy);
			}
			return levels.front().run(aDestination, aDestination, aScratch, aPolicy);
		}
	}

	template<class TSrc, class TDst>
	static bool same_layout(TSrc aSource, TDst aDestination)
	{
		if constexpr (concepts::MemoryView<TSrc> && concepts::MemoryView<TDst>) {
			const auto source_strides = aSource.strides();
			const auto destination_strides = aDestination.strides();
			return aSource.pointer() == aDestination.pointer() &&
				source_strides[0] == destination_strides[0] &&
				source_strides[1] == destination_strides[1];
		}
		return false;
	}
};

// Prepare all levels without storing any of the prototype views.
template<class TAlgorithm, bool tAnalysis>
struct multilevel_wavelet_algorithm {
	TAlgorithm transform;
	std::size_t levels;

	template<class TSrc, wavelet_subview TDst, class TPolicy>
	requires separable_wavelet_views<TSrc, TDst, TPolicy>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy aPolicy) const
	{
		const auto size = matching_dimensions(aSource, aDestination);
		check_wavelet_levels(size, levels);
		using state_t = decltype(transform.prepare(aSource, aDestination, aPolicy));
		multilevel_wavelet_state<state_t, tAnalysis> state{size, {}, 0};
		state.levels.reserve(levels);
		state.levels.push_back(transform.prepare(aSource, aDestination, aPolicy));
		state.bytes = state.levels.front().scratch_bytes();

		auto level_size = size;
		for (std::size_t i = 1; i < levels; ++i) {
			level_size = wavelet_half(level_size);
			auto ll = aDestination.subview(math::vec2i{}, level_size);
			state.levels.push_back(transform.prepare(ll, ll, aPolicy));
			state.bytes = std::max(state.bytes, state.levels.back().scratch_bytes());
		}
		return state;
	}
};

} // namespace detail

/**
 * @brief Configure a Mallat decomposition using a single-level analysis.
 * @param aForward Single-level wavelet analysis such as db2_forward.
 * @param aLevels Positive number of levels.
 * @return Algorithm for execution_context::prepare on same-backend float views.
 *
 * Each level after the first transforms only its parent's LL quadrant.
 * The destination must support subviews. Both extents must be even and
 * positive at every requested level. Exact in-place use is supported.
 * Keep the views alive through completion.
 */
template<template<int, class> class TPass>
auto multilevel_wavelet_forward(detail::separable_wavelet_algorithm<TPass, true> aForward, std::size_t aLevels)
{
	return detail::multilevel_wavelet_algorithm<decltype(aForward), true>{aForward, aLevels};
}

/**
 * @brief Configure synthesis from a packed Mallat coefficient image.
 * @param aInverse Matching single-level synthesis, such as wavelet_inverse.
 * @param aLevels Number of decomposition levels.
 * @return Algorithm for execution_context::prepare on same-backend float views.
 *
 * Restores the coarsest LL and detail bands first, then progresses toward
 * the full image. Use the original coefficient layout and wavelet.
 * Source and destination must have identical layouts for exact in-place
 * use or disjoint storage. Keep both views alive through completion.
 */
template<template<int, class> class TPass>
auto multilevel_wavelet_inverse(detail::separable_wavelet_algorithm<TPass, false> aInverse, std::size_t aLevels)
{
	return detail::multilevel_wavelet_algorithm<decltype(aInverse), false>{aInverse, aLevels};
}

/// Wavelet detail orientation, with the horizontal axis named first.
enum class wavelet_band { hl, lh, hh };

/**
 * @brief Own the view descriptors for a borrowed multilevel coefficient image.
 *
 * Each detail orientation forms a separate pyramid, finest first. Only the
 * coarsest approximation remains as an LL image. The coefficient storage must
 * outlive the views, and borrowed pyramid tables depend on this object's lifetime.
 */
template<detail::wavelet_subview TView>
class wavelet_pyramid_view {
public:
	/**
	 * @brief Describe a coefficient image produced with the given number of levels.
	 * @throws std::invalid_argument If the extents cannot support the levels.
	 */
	wavelet_pyramid_view(TView aCoefficients, std::size_t aLevels) : mCoefficients(aCoefficients)
	{
		auto size = aCoefficients.dimensions();
		detail::check_wavelet_levels(size, aLevels);
		for (auto& band : mDetails) {
			band.reserve(aLevels);
		}
		for (std::size_t i = 0; i < aLevels; ++i) {
			size = detail::wavelet_half(size);
			mDetails[0].push_back(aCoefficients.subview(math::vec2i{size[0], 0}, size));
			mDetails[1].push_back(aCoefficients.subview(math::vec2i{0, size[1]}, size));
			mDetails[2].push_back(aCoefficients.subview(size, size));
		}
		mApproximationSize = size;
	}

	/// Return the number of detail levels.
	std::size_t level_count() const { return mDetails.front().size(); }

	/// Borrow the detail pyramid for one orientation.
	pyramid_view<TView> details(wavelet_band aBand) const &
	{
		return pyramid_view<TView>(mDetails.at(static_cast<std::size_t>(aBand)));
	}
	pyramid_view<TView> details(wavelet_band) const && = delete;

	/// Return the coarsest LL view.
	TView approximation() const { return mCoefficients.subview(math::vec2i{}, mApproximationSize); }

	/// Return the full packed coefficient view for reconstruction.
	TView coefficients() const { return mCoefficients; }

private:
	TView mCoefficients;
	math::vec2i mApproximationSize;
	std::array<std::vector<TView>, 3> mDetails;
};

/// Describe a multilevel coefficient image without copying its pixels.
template<detail::wavelet_subview TView>
auto make_wavelet_pyramid_view(TView aCoefficients, std::size_t aLevels)
{
	return wavelet_pyramid_view<TView>(aCoefficients, aLevels);
}

} // namespace iane
