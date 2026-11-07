#define BOOST_TEST_MODULE iane_transform_cpu
#include <boost/test/included/unit_test.hpp>

#include "transform_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_transform<2>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_transform<3>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_transform<4>(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(mismatched_domains)
{
	using image_t = iane::host_image<int, 4>;
	image_t source(image_t::extents_t{2, 2, 2, 3});
	image_t destination(image_t::extents_t{2, 2, 2, 4});
	BOOST_CHECK_THROW(iane::transform(source.const_view(), destination.view(),
		[](int aValue) { return aValue; }), std::invalid_argument);
}
