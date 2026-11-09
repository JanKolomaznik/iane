#define BOOST_TEST_MODULE iane_color_morphology_cpu
#include <boost/test/included/unit_test.hpp>

#include "color_morphology_cases.hpp"

BOOST_AUTO_TEST_CASE(color_conversions) { meta_tests::check_color_conversions(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(flat_morphology) { meta_tests::check_flat_morphology(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(morphology_box_in_two_dimensions)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<float, 2> input(3, 3);
	iane::host_image<float, 2> output(3, 3);
	iane::copy(iane::constant_view(input.view(), 0.0f), input.view());
	const coord_t center{1, 1};
	input.view()[center] = 7.0f;
	iane::dilate(input.const_view(), output.view(), iane::box_footprint(coord_t{1, 1}));
	for (int i = 0; i < 9; ++i) {
		BOOST_TEST(output.pointer()[i] == 7.0f);
	}
	BOOST_CHECK_THROW(iane::erode(input.const_view(), output.view(), iane::box_footprint(coord_t{-1, 1})),
		std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(opening_and_closing_on_strided_4d_views)
{
	meta_tests::check_compound_morphology(iane::backend::cpu);
}

BOOST_AUTO_TEST_CASE(opening_scalar_radius_and_reuse)
{
	iane::host_image<int, 1> image(3);
	image.view()[0] = 1;
	image.view()[1] = 8;
	image.view()[2] = 1;
	auto exec = iane::make_execution_context();
	auto opening = exec.prepare(iane::opening(iane::box_footprint(1)), image.const_view(), image.view());
	exec.run(opening, image.const_view(), image.view()).wait();
	BOOST_TEST(image.view()[0] == 1);
	BOOST_TEST(image.view()[1] == 1);
	BOOST_TEST(image.view()[2] == 1);
	auto shorter = image.view().subview(0, 2);
	BOOST_CHECK_THROW(exec.run(opening, shorter, shorter), std::invalid_argument);
}
