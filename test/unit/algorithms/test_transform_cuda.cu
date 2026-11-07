#define BOOST_TEST_MODULE iane_transform_cuda
#include "cuda_test_main.hpp"

#include "transform_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_transform<2>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_transform<3>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_transform<4>(iane::backend::cuda); }
