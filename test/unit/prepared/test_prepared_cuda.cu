#define BOOST_TEST_MODULE iane_prepared_cuda
#include "../algorithms/cuda_test_main.hpp"

#include "prepared_cases.hpp"

BOOST_AUTO_TEST_CASE(preparation_runs_once) { prepared_tests::check_prepared_runs(iane::backend::cuda); }

BOOST_AUTO_TEST_CASE(haar_coefficients) { prepared_tests::check_haar(iane::backend::cuda); }

#if defined(IANE_TEST_FFT)
BOOST_AUTO_TEST_CASE(fft_coefficients_and_round_trip) { prepared_tests::check_fft(iane::backend::cuda); }
#endif
