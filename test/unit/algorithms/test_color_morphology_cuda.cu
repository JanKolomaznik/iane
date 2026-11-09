#define BOOST_TEST_MODULE iane_color_morphology_cuda
#include "cuda_test_main.hpp"

#include "color_morphology_cases.hpp"

BOOST_AUTO_TEST_CASE(color_conversions) { meta_tests::check_color_conversions(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(flat_morphology) { meta_tests::check_flat_morphology(iane::backend::cuda); }

BOOST_AUTO_TEST_CASE(opening_and_closing_on_strided_4d_views)
{
	meta_tests::check_compound_morphology(iane::backend::cuda);
}
