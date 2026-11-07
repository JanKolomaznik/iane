#pragma once

#include "algorithm_test_views.hpp"

#include <iane/algorithms/for_each.hpp>

namespace meta_tests {

// Initialize pixels from coordinates for the position visitor.
struct initialize_pixel {
	template<class TPosition>
	IANE_DECL_HYBRID void operator()(pixel& aPixel, const TPosition& aPosition) const
	{
		aPixel = pattern(aPosition);
	}
};

// Mutate the pixel supplied by the value visitor.
struct double_pixel {
	IANE_DECL_HYBRID void operator()(pixel& aPixel) const { aPixel = aPixel * 2; }
};

// Exercise a locator callback that writes to a separate destination.
template<class TView>
struct store_previous_sample {
	TView output;

	IANE_CUDA_HD_WARNING_DISABLE
	template<iane::concepts::Locator TLocator>
	IANE_DECL_HYBRID void operator()(TLocator aLocator) const
	{
		typename TLocator::offset_t offset{};
		offset[TView::dimension - 1] = -1;
		output[aLocator.position()] = aLocator[offset];
	}
};

template<int tDim>
void check_for_each(iane::backend aBackend)
{
	const auto input = make_input<tDim>();
	iane::host_image<pixel, tDim> output(input.dimensions());
	const iane::run_options options{.backend = aBackend};

	iane::run_on(options, input.const_view(), output.view(), [](auto, auto aDst, auto& aExec) {
		iane::for_each_position(aDst, initialize_pixel{}, aExec);
		return iane::for_each(aDst, double_pixel{}, aExec);
	});
	check_pixels(output, [](int aValue, int, int) {
		return pixel{2 * aValue, -2 * aValue};
	});

	iane::run_on(options, input.const_view(), output.view(), [](auto, auto aDst, auto& aExec) {
		return iane::for_each_locator(make_pattern(aDst), store_previous_sample{aDst},
			iane::border_mode::zero, aExec);
	});
	check_pixels(output, [](int aValue, int aLayer, int aPlaneSize) {
		const int previous = aLayer == 0 ? 0 : aValue - aPlaneSize;
		return pixel{previous, -previous};
	});
}

} // namespace meta_tests
