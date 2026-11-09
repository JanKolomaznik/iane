#define BOOST_TEST_MODULE iane_resample_cpu
#include <boost/test/included/unit_test.hpp>

#include "resample_cases.hpp"

BOOST_AUTO_TEST_CASE(resize_and_sample_neighborhood)
{
	resample_tests::check_resampling(iane::backend::cpu);
}

BOOST_AUTO_TEST_CASE(resample_volume_into_plane)
{
	resample_tests::check_plane(iane::backend::cpu);
}

BOOST_AUTO_TEST_CASE(nearest_uses_outer_corner_coordinates)
{
	iane::host_image<int, 1> line(2);
	line.view()[0] = 10;
	line.view()[1] = 20;
	const auto nearest = iane::make_interpolated_view(line.const_view(), iane::nearest, iane::border_mode::zero);
	BOOST_TEST(nearest[1] == 20);
	BOOST_TEST(nearest.sample(-0.01f) == 0);
	BOOST_TEST(nearest.sample(0.0f) == 10);
	BOOST_TEST(nearest.sample(0.75f) == 10);
	BOOST_TEST(nearest.sample(1.0f) == 20);
	BOOST_TEST(nearest.sample(2.0f) == 0);

	const iane::sampling_image_locator locator(nearest, 0.75f);
	BOOST_TEST(locator.position() == 0.75f);
	BOOST_TEST(locator.sample(0.25f) == 20);
	const auto linear = iane::make_interpolated_view(line.const_view(), iane::linear);
	BOOST_TEST(linear.sample(-1.0f) == 10.0f);
	BOOST_TEST(linear.sample(1.0f) == 15.0f);
	BOOST_TEST(linear.sample(3.0f) == 20.0f);
	BOOST_TEST(linear.sample(std::numeric_limits<float>::infinity()) == 0.0f);
}
