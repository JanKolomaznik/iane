#define BOOST_TEST_MODULE iane_convolution_cuda
#include "cuda_test_main.hpp"

#include "convolution_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_convolution<2>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_convolution<3>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_convolution<4>(iane::backend::cuda); }
