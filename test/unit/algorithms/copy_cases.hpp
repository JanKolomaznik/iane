#pragma once

#include "algorithm_test_views.hpp"

#include <iane/algorithms/copy.hpp>
#include <iane/view/algebra.hpp>

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

// Generate a coordinate-dependent offset for a 2D expression.
struct column_index {
	IANE_DECL_HYBRID int operator()(const iane::math::vec2i& aPosition) const { return aPosition[0]; }
};

// Double an input pixel to make the unary adapter observable.
struct double_pixel {
	IANE_DECL_HYBRID int operator()(int aPixel) const { return 2 * aPixel; }
};

// Combine two operands without requiring device support from std::plus.
struct add_pixels {
	IANE_DECL_HYBRID int operator()(int aLeft, int aRight) const { return aLeft + aRight; }
};

inline void check_procedural_copy(iane::backend aBackend)
{
	using image_t = iane::host_image<int, 2>;
	using coord_t = image_t::extents_t;
	image_t input(coord_t{3, 2});
	image_t output(coord_t{2, 2});
	for (int i = 0; i < 6; ++i) {
		input.pointer()[i] = i + 1;
	}
	const auto source = input.const_view().subview(coord_t{1, 0}, coord_t{2, 2});
	iane::run_on(iane::run_options{.backend = aBackend}, source, output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			auto mapped = iane::map_view(aSource, double_pixel{});
			auto generated = iane::generate_view(aSource, column_index{});
			auto constant = iane::constant_view(aSource, 1);
			auto bias = iane::zip_view(generated, constant, add_pixels{});
			auto expression = iane::zip_view(mapped, bias, add_pixels{});
			static_assert(iane::concepts::ReadableView<decltype(expression)>);
			static_assert(!iane::concepts::MemoryView<decltype(expression)>);
			static_assert(iane::concepts::SameViewBackend<decltype(expression), decltype(aSource)>);
			return iane::copy(expression, aDestination, aExecution);
		});
	const int expected[]{5, 8, 11, 14};
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(output.pointer()[i] == expected[i]);
	}
}

inline void check_view_algebra(iane::backend aBackend)
{
	iane::host_image<int, 1> input(3);
	iane::host_image<int, 1> output(3);
	input.pointer()[0] = 2;
	input.pointer()[1] = 3;
	input.pointer()[2] = 5;
	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			auto expression = (aSource * 2 + 1) + (20 / aSource) + -(10 - aSource);
			return iane::copy(expression, aDestination, aExecution);
		});
	const int expected[]{7, 6, 10};
	for (int i = 0; i < 3; ++i) {
		BOOST_TEST(output.pointer()[i] == expected[i]);
	}
}

} // namespace meta_tests
