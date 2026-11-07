#define BOOST_TEST_MODULE iane_host_meta_algorithms
#include <boost/test/included/unit_test.hpp>

#include "meta_algorithm_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_meta_algorithms<2>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_meta_algorithms<3>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_meta_algorithms<4>(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(empty_and_mismatched_domains)
{
	using image_t = iane::host_image<int, 4>;
	image_t empty(image_t::extents_t{2, 2, 0, 3});
	int visits = 0;
	iane::for_each(empty.view(), [&](int&) { ++visits; });
	BOOST_TEST(visits == 0);

	image_t source(image_t::extents_t{2, 2, 2, 3});
	image_t destination(image_t::extents_t{2, 2, 2, 4});
	BOOST_CHECK_THROW(iane::transform(source.const_view(), destination.view(),
		[](int aValue) { return aValue; }), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(scalar_coordinates)
{
	iane::host_image<int, 1> source(3);
	iane::host_image<int, 1> kernel(3);
	iane::host_image<int, 1> output(3);
	iane::for_each_position(source.view(), [](int& aValue, int aPosition) { aValue = aPosition + 1; });
	kernel.pointer()[0] = 1;
	kernel.pointer()[1] = 0;
	kernel.pointer()[2] = 0;
	iane::convolve(source.const_view(), kernel.const_view(), output.view(), {iane::border_mode::zero});
	BOOST_TEST(output.pointer()[0] == 2);
	BOOST_TEST(output.pointer()[1] == 3);
	BOOST_TEST(output.pointer()[2] == 0);
}

BOOST_AUTO_TEST_CASE(convolution_flips_kernel_and_respects_crop_boundary)
{
	using namespace iane;
	using coord_t = math::dim_traits<2>::coord_t;
	iane::host_image<std::uint8_t, 2> source(4, 2);
	iane::host_image<std::uint8_t, 2> result(2, 2);
	iane::host_image<float, 2> kernel(3, 1);
	for_each_position(source.view(), [](std::uint8_t& aValue, const coord_t& aPosition) {
		aValue = static_cast<std::uint8_t>(10 * aPosition[0] + 100 * aPosition[1]);
	}).wait();
	kernel.pointer()[0] = 1.0f;
	kernel.pointer()[1] = 0.0f;
	kernel.pointer()[2] = 0.0f;

	const auto crop = source.const_view().subview(coord_t{1, 0}, coord_t{2, 2});
	convolve(crop, kernel.const_view(), result.view(), {border_mode::clamp}).wait();
	BOOST_TEST(+result.pointer()[0] == 20);
	BOOST_TEST(+result.pointer()[1] == 20);
	BOOST_TEST(+result.pointer()[2] == 120);
	BOOST_TEST(+result.pointer()[3] == 120);

	convolve(crop, kernel.const_view(), result.view(), {border_mode::zero}).wait();
	BOOST_TEST(+result.pointer()[0] == 20);
	BOOST_TEST(+result.pointer()[1] == 0);
	BOOST_TEST(+result.pointer()[2] == 120);
	BOOST_TEST(+result.pointer()[3] == 0);
}
