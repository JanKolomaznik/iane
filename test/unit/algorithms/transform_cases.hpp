#pragma once

#include "algorithm_test_views.hpp"

#include <iane/algorithms/transform.hpp>

namespace meta_tests {

// Add a fixed bias through the value transform.
struct bias_pixel {
	IANE_DECL_HYBRID pixel operator()(pixel aPixel) const { return aPixel + pixel{1, -1}; }
};

// Add a bias derived from the outermost coordinate.
template<class TView>
struct combine_at_position {
	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID pixel operator()(pixel aPixel, const typename TView::coord_t& aPosition) const
	{
		return aPixel + pixel{aPosition[TView::dimension - 1], 0};
	}
};

// Sample the neighbor along the outermost axis to exercise higher dimensions.
struct next_sample {
	IANE_CUDA_HD_WARNING_DISABLE
	template<iane::concepts::Locator TLocator>
	IANE_DECL_HYBRID pixel operator()(TLocator aLocator) const
	{
		typename TLocator::offset_t offset{};
		offset[TLocator::dimension - 1] = 1;
		return aLocator[offset];
	}
};

template<int tDim>
void check_transform(iane::backend aBackend)
{
	const auto input = make_input<tDim>();
	iane::host_image<pixel, tDim> output(input.dimensions());
	const iane::run_options options{.backend = aBackend};

	iane::run_on(options, input.const_view(), output.view(), [](auto aSrc, auto aDst, auto& aExec) {
		iane::transform(aSrc, aDst, bias_pixel{}, aExec);
		return iane::transform(aDst, aDst, bias_pixel{}, aExec);
	});
	check_pixels(output, [](int aValue, int, int) {
		return pixel{aValue + 2, -aValue - 2};
	});

	iane::run_on(options, input.const_view(), output.view(), [](auto aSrc, auto aDst, auto& aExec) {
		return iane::transform_position(aSrc, aDst, combine_at_position<decltype(aDst)>{}, aExec);
	});
	check_pixels(output, [](int aValue, int aLayer, int) {
		return pixel{aValue + aLayer, -aValue};
	});

	iane::run_on(options, input.const_view(), output.view(), [](auto aSrc, auto aDst, auto& aExec) {
		return iane::transform_locator(aSrc, aDst, next_sample{}, iane::border_mode::clamp, aExec);
	});
	check_pixels(output, [](int aValue, int aLayer, int aPlaneSize) {
		const int next = aLayer == 2 ? aValue : aValue + aPlaneSize;
		return pixel{next, -next};
	});
}

} // namespace meta_tests
