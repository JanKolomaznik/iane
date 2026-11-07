#define BOOST_TEST_MODULE iane_for_each_cpu
#include <boost/test/included/unit_test.hpp>

#include "for_each_cases.hpp"

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
