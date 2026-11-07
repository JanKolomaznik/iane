#define BOOST_TEST_MODULE iane_prepared_cpu
#include <boost/test/included/unit_test.hpp>

#include "prepared_cases.hpp"

BOOST_AUTO_TEST_CASE(preparation_runs_once) { prepared_tests::check_prepared_runs(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(haar_coefficients) { prepared_tests::check_haar(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(haar_in_place_strided_and_context_validation)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<float, 2> storage(coord_t{5, 4});
	auto crop = storage.view().subview(coord_t{1, 1}, coord_t{2, 2});
	iane::for_each(storage.view(), [](float& aPixel) { aPixel = 42.0f; });
	iane::for_each(crop, [](float& aPixel) { aPixel = 3.0f; });
	auto context = iane::make_execution_context(iane::cpu_policy{.threads = 2, .grain_size = 1});
	auto other = iane::make_execution_context();
	auto haar = context.prepare(iane::wavelet_forward, crop, crop);
	BOOST_CHECK_THROW(other.run(haar, crop, crop), std::invalid_argument);
	context.run(haar, crop, crop).wait();
	const coord_t dc{0, 0};
	const coord_t high{1, 1};
	const coord_t outside{0, 0};
	BOOST_TEST(crop[dc] == 6.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_TEST(crop[high] == 0.0f);
	BOOST_TEST(storage.const_view()[outside] == 42.0f);
	BOOST_CHECK_THROW(context.prepare(iane::wavelet_forward, storage.view(), storage.view()), std::invalid_argument);
}

#if defined(IANE_TEST_FFT)
BOOST_AUTO_TEST_CASE(fft_coefficients_and_round_trip) { prepared_tests::check_fft(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(fft_reuses_plan_with_new_buffers_and_rejects_aliasing)
{
	using coord_t = iane::math::vec2i;
	using image_t = iane::host_image<iane::fft_complex, 2>;
	const coord_t size{4, 2};
	image_t prototype_in(size);
	image_t prototype_out(size);
	image_t input(size);
	image_t output(size);
	iane::for_each(input.view(), [](iane::fft_complex& aPixel) { aPixel = {2.0f, 0.0f}; });
	auto context = iane::make_execution_context();
	auto fft = context.prepare(iane::fft_forward, prototype_in.const_view(), prototype_out.view());
	BOOST_CHECK_THROW(context.run(fft, input.const_view(), input.view()), std::invalid_argument);
	context.run(fft, input.const_view(), output.view()).wait();
	BOOST_TEST(output.pointer()[0].real == 16.0f);
	for (int i = 1; i < 8; ++i) {
		BOOST_TEST(output.pointer()[i].real == 0.0f);
		BOOST_TEST(output.pointer()[i].imag == 0.0f);
	}
	BOOST_TEST(context.scratch_capacity() == 0u);
}
#endif
