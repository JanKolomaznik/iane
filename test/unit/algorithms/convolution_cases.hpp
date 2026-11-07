#pragma once

#include "algorithm_test_views.hpp"

#include <iane/algorithms/convolution.hpp>

namespace meta_tests {

static_assert(iane::concepts::ConvolutionElements<pixel, int, pixel>);
static_assert(!iane::concepts::ConvolutionElements<pixel, pixel, pixel>);

template<int tDim>
void check_convolution(iane::backend aBackend)
{
	const auto input = make_input<tDim>();
	iane::host_image<pixel, tDim> output(input.dimensions());
	const iane::run_options options{.backend = aBackend};

	using coord_t = typename decltype(output)::extents_t;
	coord_t kernel_size(1, std::true_type{});
	kernel_size[tDim - 1] = 3;
	iane::host_image<int, tDim> kernel(kernel_size);
	kernel.pointer()[0] = 2;
	kernel.pointer()[1] = -1;
	kernel.pointer()[2] = 0;
	iane::run_on(options, std::tuple{input.const_view(), kernel.const_view()}, output.view(),
		[](auto aSrc, auto aKernel, auto aDst, auto& aExec) {
			return iane::convolve(aSrc, aKernel, aDst, {iane::border_mode::zero}, aExec);
		});
	check_pixels(output, [](int aValue, int aLayer, int aPlaneSize) {
		const int next = aLayer == 2 ? 0 : aValue + aPlaneSize;
		return pixel{2 * next - aValue, aValue - 2 * next};
	});
}

} // namespace meta_tests
