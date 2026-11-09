#pragma once

#include <iane/algorithms/color/conversion.hpp>
#include <iane/algorithms/morphology/morphology.hpp>
#include <iane/execution/run_on.hpp>
#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>
#include <iane/view/procedural.hpp>

#include <boost/test/unit_test.hpp>

namespace meta_tests {

// Exercise byte-channel normalization with an ordinary pixel type.
struct color_pixel {
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
};

inline void check_color_conversions(iane::backend aBackend)
{
	using color_t = iane::math::vec3f;
	iane::host_image<color_pixel, 1> input(4);
	iane::host_image<color_t, 1> output(4);
	input.view()[0] = {255, 255, 255};
	input.view()[1] = {255, 0, 0};
	input.view()[2] = {128, 128, 128};
	input.view()[3] = {32, 119, 210};
	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto encoded = iane::map_view(aSource, iane::rgb8_to_unit);
			const auto linear = iane::map_view(encoded, iane::srgb_to_linear);
			const auto xyz = iane::map_view(linear, iane::linear_rgb_to_xyz);
			return iane::copy(iane::map_view(xyz, iane::xyz_to_lab), aDestination, aExecution);
		});
	BOOST_CHECK_SMALL(output.view()[0][0] - 100.0f, 0.0002f);
	BOOST_CHECK_SMALL(output.view()[0][1], 0.0002f);
	BOOST_CHECK_SMALL(output.view()[0][2], 0.0002f);
	BOOST_CHECK_SMALL(output.view()[1][0] - 53.2371f, 0.001f);
	BOOST_CHECK_SMALL(output.view()[1][1] - 80.0901f, 0.001f);
	BOOST_CHECK_SMALL(output.view()[1][2] - 67.2033f, 0.001f);
	BOOST_CHECK_SMALL(output.view()[2][0] - 53.5850f, 0.001f);

	iane::host_image<color_t, 1> restored(4);
	iane::run_on(iane::run_options{.backend = aBackend}, output.const_view(), restored.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto xyz = iane::map_view(aSource, iane::lab_to_xyz);
			const auto linear = iane::map_view(xyz, iane::xyz_to_linear_rgb);
			const auto encoded = iane::map_view(linear, iane::linear_to_srgb);
			return iane::copy(encoded, aDestination, aExecution);
		});
	for (int i = 0; i < 4; ++i) {
		const auto expected = iane::rgb8_to_unit(input.view()[i]);
		for (int c = 0; c < 3; ++c) {
			BOOST_CHECK_SMALL(restored.view()[i][c] - expected[c], 0.00002f);
		}
	}

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto hsv = iane::map_view(iane::map_view(aSource, iane::rgb8_to_unit), iane::rgb_to_hsv);
			return iane::copy(hsv, aDestination, aExecution);
		});
	BOOST_TEST(output.view()[0][0] == 0.0f);
	BOOST_TEST(output.view()[0][1] == 0.0f);
	BOOST_TEST(output.view()[1][0] == 0.0f);
	BOOST_TEST(output.view()[1][1] == 1.0f);
	BOOST_CHECK_SMALL(output.view()[3][0] - 0.5852f, 0.001f);
	iane::run_on(iane::run_options{.backend = aBackend}, output.const_view(), restored.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			return iane::copy(iane::map_view(aSource, iane::hsv_to_rgb), aDestination, aExecution);
		});
	for (int i = 0; i < 4; ++i) {
		const auto expected = iane::rgb8_to_unit(input.view()[i]);
		for (int c = 0; c < 3; ++c) {
			BOOST_CHECK_SMALL(restored.view()[i][c] - expected[c], 0.00002f);
		}
	}
}

// Select the positive offset to expose reflection for dilation.
struct positive_neighbor {
	IANE_DECL_HYBRID bool operator()(int aOffset) const { return aOffset == 1; }
};

inline void check_flat_morphology(iane::backend aBackend)
{
	iane::host_image<int, 1> input(6);
	iane::host_image<int, 1> output(4);
	const int values[]{99, 3, 1, 5, 2, 99};
	std::copy_n(values, 6, input.pointer());
	for (const bool dilate : {false, true}) {
		for (const bool asymmetric : {false, true}) {
			iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
				[dilate, asymmetric](auto aSource, auto aDestination, auto aExecution) {
					const auto crop = aSource.subview(1, 4);
					if (asymmetric) {
						const auto footprint = iane::make_flat_footprint(1, positive_neighbor{});
						const iane::morphology_options options{.border = iane::border_mode::zero};
						return dilate ? iane::dilate(crop, aDestination, footprint, options, aExecution) :
							iane::erode(crop, aDestination, footprint, options, aExecution);
					}
					const auto footprint = iane::box_footprint(1);
					return dilate ? iane::dilate(crop, aDestination, footprint, {}, aExecution) :
						iane::erode(crop, aDestination, footprint, {}, aExecution);
				});
			const int minimum[]{1, 1, 1, 2};
			const int maximum[]{3, 5, 5, 5};
			const int forward[]{1, 5, 2, 0};
			const int backward[]{0, 3, 1, 5};
			const auto* expected = asymmetric ? (dilate ? backward : forward) : (dilate ? maximum : minimum);
			for (int i = 0; i < 4; ++i) {
				BOOST_TEST(output.view()[i] == expected[i]);
			}
		}
	}
}

inline void check_compound_morphology(iane::backend aBackend)
{
	using coord_t = iane::math::dim_traits<4>::coord_t;
	const coord_t storage_size{6, 3, 3, 3};
	const coord_t size{4, 3, 3, 3};
	const coord_t origin{1, 0, 0, 0};
	iane::host_image<int, 4> input(storage_size);
	iane::host_image<int, 4> output(storage_size);
	for (const bool close : {false, true}) {
		iane::copy(iane::constant_view(input.view(), 99), input.view());
		const auto crop = input.view().subview(origin, size);
		iane::for_each_position(crop, [close](int& aPixel, coord_t aPosition) {
			const bool plane = aPosition[0] < 2;
			const bool impulse = aPosition[0] == 3 && aPosition[1] == 1 &&
				aPosition[2] == 1 && aPosition[3] == 1;
			aPixel = close ? !(plane || impulse) : (plane || impulse);
		});
		iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
			[=](auto aSource, auto aDestination, auto aExecution) {
				const auto policy = static_cast<iane::view_policy<decltype(aDestination)>>(aExecution);
				auto exec = iane::make_execution_context(policy);
				const auto source = aSource.subview(origin, size);
				const auto destination = aDestination.subview(origin, size);
				const auto box = iane::box_footprint(coord_t{1, 1, 1, 1});
				iane::copy(iane::constant_view(aDestination, 99), aDestination, policy);
				if (close) {
					auto plan = exec.prepare(iane::closing(box), destination, destination);
					iane::copy(source, destination, policy);
					exec.run(plan, destination, destination).wait();
				} else {
					auto plan = exec.prepare(iane::opening(box), source, destination);
					exec.run(plan, source, destination).wait();
				}
				BOOST_TEST(exec.scratch_capacity() == 108 * sizeof(int));
				return aExecution;
			});
		iane::for_each_position(output.const_view().subview(origin, size),
			[close](int aPixel, coord_t aPosition) {
				const int expected = close ? aPosition[0] >= 2 : aPosition[0] < 2;
				BOOST_TEST(aPixel == expected);
			});
		const coord_t outside{};
		BOOST_TEST(output.const_view()[outside] == 99);
	}
}

} // namespace meta_tests
