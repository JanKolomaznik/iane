#define BOOST_TEST_MODULE iane_for_each_cuda
#include "cuda_test_main.hpp"

#include "for_each_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_for_each<2>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_for_each<3>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_for_each<4>(iane::backend::cuda); }
