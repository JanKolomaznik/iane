#define BOOST_TEST_MODULE iane_for_each_cpu
#include <boost/test/included/unit_test.hpp>

#include "for_each_cases.hpp"

#include <set>

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_for_each<2>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_for_each<3>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_for_each<4>(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(empty_domain)
{
	using image_t = iane::host_image<int, 4>;
	image_t empty(image_t::extents_t{2, 2, 0, 3});
	int visits = 0;
	iane::for_each(empty.view(), [&](int&) { ++visits; });
	BOOST_TEST(visits == 0);
}

BOOST_AUTO_TEST_CASE(parallel_policy_visits_each_strided_pixel_once)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<int, 2> image(coord_t{7, 5});
	iane::for_each(image.view(), [](int& aPixel) { aPixel = 0; });
	auto crop = image.view().subview(coord_t{1, 1}, coord_t{5, 3});
	std::thread::id workers[15];
	iane::for_each_position(crop, [&](int& aPixel, const coord_t& aPosition) {
		++aPixel;
		workers[aPosition[0] + 5 * aPosition[1]] = std::this_thread::get_id();
	}, iane::cpu_policy{.threads = 3, .grain_size = 4}).wait();

	for (int y = 0; y < 5; ++y) {
		for (int x = 0; x < 7; ++x) {
			const coord_t position{x, y};
			BOOST_TEST(image.const_view()[position] == (x >= 1 && x <= 5 && y >= 1 && y <= 3 ? 1 : 0));
		}
	}
	const std::set<std::thread::id> distinct_workers(std::begin(workers), std::end(workers));
	BOOST_TEST(distinct_workers.size() == 3u);
}

BOOST_AUTO_TEST_CASE(cpu_policy_errors_reach_the_caller)
{
	iane::host_image<int, 1> image(17);
	iane::host_image<int, 1> output(17);
	BOOST_CHECK_THROW(iane::copy(image.const_view(), output.view(),
		iane::cpu_policy{.threads = 0}), std::invalid_argument);
	BOOST_CHECK_THROW(iane::for_each(image.view(), [](int&) {},
		iane::cpu_policy{.grain_size = 0}), std::invalid_argument);
	BOOST_CHECK_THROW(iane::for_each_position(image.view(), [](int&, int aPosition) {
		if (aPosition == 16) {
			throw std::logic_error("worker failure");
		}
	}, (iane::cpu_policy{.threads = 3, .grain_size = 4})), std::logic_error);
}
