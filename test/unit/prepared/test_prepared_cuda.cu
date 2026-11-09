#define BOOST_TEST_MODULE iane_prepared_cuda
#include "../algorithms/cuda_test_main.hpp"

#include "spectral_cases.hpp"

#if defined(IANE_TEST_FFT)
#include "real_fft_cases.hpp"
#endif
#include "composition_cases.hpp"

BOOST_AUTO_TEST_CASE(preparation_runs_once) { prepared_tests::check_prepared_runs(iane::backend::cuda); }

BOOST_AUTO_TEST_CASE(haar_coefficients) { prepared_tests::check_haar(iane::backend::cuda); }

#if defined(IANE_TEST_FFT)
BOOST_AUTO_TEST_CASE(fft_coefficients_and_round_trip) { prepared_tests::check_fft(iane::backend::cuda); }
#endif

BOOST_AUTO_TEST_CASE(scan_reduce_and_integral_image) { prepared_tests::check_sequences(iane::backend::cuda); }

BOOST_AUTO_TEST_CASE(histogram_equalization_and_stable_compaction)
{
	prepared_tests::check_histogram_and_compaction(iane::backend::cuda);
}

BOOST_AUTO_TEST_CASE(spectral_display_views) { prepared_tests::check_spectrum_views(iane::backend::cuda); }

BOOST_AUTO_TEST_CASE(additional_wavelets) { prepared_tests::check_additional_wavelets(iane::backend::cuda); }

#if defined(IANE_TEST_FFT)
BOOST_AUTO_TEST_CASE(real_fft_odd_and_even_round_trips)
{
	using coord_t = iane::math::vec2i;
	prepared_tests::check_real_fft(iane::backend::cuda, coord_t{4, 3});
	prepared_tests::check_real_fft(iane::backend::cuda, coord_t{5, 3});
	prepared_tests::check_real_fft(iane::backend::cuda, coord_t{6, 4});
}
#endif
