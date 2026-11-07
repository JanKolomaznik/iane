#pragma once

#include "algorithm_test_views.hpp"

#include <iane/algorithms/copy.hpp>

namespace meta_tests {

template<int tDim>
void check_copy(iane::backend aBackend)
{
	const auto input = make_input<tDim>();
	iane::host_image<pixel, tDim> output(input.dimensions());
	const iane::run_options options{.backend = aBackend};

	iane::run_on(options, input.const_view(), output.view(), [](auto aSrc, auto aDst, auto& aExec) {
		return iane::copy(make_pattern(aSrc), aDst, aExec);
	});
	check_pixels(output, [](int aValue, int, int) { return pixel{aValue, -aValue}; });
}

} // namespace meta_tests
