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

// Execute the prepared first level, then reuse its workspace in nested LL views.
template<class TState>
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

		auto first_workspace = aScratch;
		auto result = levels.front().run(aSource, aDestination, first_workspace, aPolicy);
		auto level_size = size;
		for (std::size_t i = 1; i < levels.size(); ++i) {
			level_size = wavelet_half(level_size);
			auto ll = aDestination.subview(math::vec2i{}, level_size);
			auto workspace = aScratch;
			result = levels[i].run(ll, ll, workspace, aPolicy);
		}
		return result;
	}
};

// Prepare all levels without storing any of the prototype views.
template<class TAlgorithm>
struct multilevel_wavelet_algorithm {
	TAlgorithm forward;
	std::size_t levels;

	template<class TSrc, wavelet_subview TDst, class TPolicy>
	requires separable_wavelet_views<TSrc, TDst, TPolicy>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy aPolicy) const
	{
		const auto size = matching_dimensions(aSource, aDestination);
		check_wavelet_levels(size, levels);
		using state_t = decltype(forward.prepare(aSource, aDestination, aPolicy));
		multilevel_wavelet_state<state_t> state{size, {}, 0};
		state.levels.reserve(levels);
		state.levels.push_back(forward.prepare(aSource, aDestination, aPolicy));
		state.bytes = state.levels.front().scratch_bytes();

		auto level_size = size;
		for (std::size_t i = 1; i < levels; ++i) {
			level_size = wavelet_half(level_size);
			auto ll = aDestination.subview(math::vec2i{}, level_size);
			state.levels.push_back(forward.prepare(ll, ll, aPolicy));
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
	return detail::multilevel_wavelet_algorithm{aForward, aLevels};
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
	wavelet_pyramid_view(TView aCoefficients, std::size_t aLevels) : mApproximation(aCoefficients)
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
		mApproximation = aCoefficients.subview(math::vec2i{}, size);
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
	TView approximation() const { return mApproximation; }

private:
	TView mApproximation;
	std::array<std::vector<TView>, 3> mDetails;
};

/// Describe a multilevel coefficient image without copying its pixels.
template<detail::wavelet_subview TView>
auto make_wavelet_pyramid_view(TView aCoefficients, std::size_t aLevels)
{
	return wavelet_pyramid_view<TView>(aCoefficients, aLevels);
}

} // namespace iane
