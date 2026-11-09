#pragma once

#include <iane/algorithms/wavelet/haar.hpp>
#include <iane/execution/run_on.hpp>
#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>

#include <boost/test/unit_test.hpp>

#if defined(IANE_TEST_FFT)
#include <iane/algorithms/fft/fft.hpp>
#endif

namespace prepared_tests {

inline void check_prepared_runs(iane::backend aBackend)
{
	iane::host_image<int, 2> output(iane::math::vec2i{2, 2});
	int preparations = 0;
	int calls = 0;
	iane::run_prepared_on(iane::run_options{.backend = aBackend, .warmup = 2, .runs = 3},
		std::tuple{}, output.view(), [&](auto aDestination, auto aExecution) {
			++preparations;
			return [&, aDestination, aExecution] {
				return iane::copy(iane::constant_view(aDestination, ++calls), aDestination, aExecution);
			};
		});
	BOOST_TEST(preparations == 1);
	BOOST_TEST(calls == 5);
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(output.pointer()[i] == 5);
	}
}

inline void check_haar(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<float, 2> input(coord_t{4, 2});
	iane::host_image<float, 2> output(coord_t{4, 2});
	for (int i = 0; i < 8; ++i) {
		input.pointer()[i] = static_cast<float>(i + 1);
	}
	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto policy = static_cast<iane::view_policy<decltype(aDestination)>>(aExecution);
			auto context = iane::make_execution_context(policy);
			auto haar = context.prepare(iane::wavelet_forward, aSource, aDestination);
			const auto capacity = context.scratch_capacity();
			BOOST_TEST(capacity == 8 * sizeof(float));
			context.run(haar, aSource, aDestination);
			const auto result = context.run(haar, aSource, aDestination);
			result.wait();
			BOOST_TEST(context.scratch_capacity() == capacity);
			return result;
		});
	const float expected[]{7, 11, -1, -1, -4, -4, 0, 0};
	for (int i = 0; i < 8; ++i) {
		BOOST_TEST(output.pointer()[i] == expected[i], boost::test_tools::tolerance(0.00001f));
	}
}

#if defined(IANE_TEST_FFT)
template<class TElement, class TView>
using backend_image = std::conditional_t<iane::concepts::HostView<TView>,
	iane::host_image<TElement, 2>,
#if defined(__CUDACC__)
	iane::device_image<TElement, 2>
#else
	iane::host_image<TElement, 2>
#endif
>;

inline void check_fft(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	using image_t = iane::host_image<iane::fft_complex, 2>;
	const coord_t size{4, 2};
	image_t input(size);
	image_t output(size);
	for (int i = 0; i < 8; ++i) {
		input.pointer()[i] = {0.0f, 0.0f};
	}
	input.pointer()[1] = {1.0f, 0.0f};

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[size](auto aSource, auto aDestination, auto aExecution) {
			using output_t = decltype(aDestination);
			const auto policy = static_cast<iane::view_policy<output_t>>(aExecution);
			backend_image<iane::fft_complex, output_t> restored(size);
			backend_image<float, output_t> bands(size);
			auto context = iane::make_execution_context(policy);
			auto forward = context.prepare(iane::fft_forward, aSource, aDestination);
			auto inverse = context.prepare(iane::fft_inverse, aDestination, restored.view());
			context.run(forward, aSource, aDestination);
			auto constant = iane::constant_view(bands.view(), 2.0f);
			auto haar = context.prepare(iane::wavelet_forward, constant, bands.view());
			const auto capacity = context.scratch_capacity();
			context.run(haar, constant, bands.view());
			context.run(inverse, aDestination, restored.view());
			context.run(forward, aSource, aDestination);

			iane::host_image<float, 2> coefficients(size);
			iane::copy(bands.const_view(), coefficients.view(), policy).wait();
			BOOST_TEST(coefficients.pointer()[0] == 4.0f, boost::test_tools::tolerance(0.00001f));
			BOOST_TEST(coefficients.pointer()[2] == 0.0f);
			image_t round_trip(size);
			iane::copy(restored.const_view(), round_trip.view(), policy).wait();
			for (int i = 0; i < 8; ++i) {
				BOOST_TEST(round_trip.pointer()[i].real == (i == 1 ? 8.0f : 0.0f),
					boost::test_tools::tolerance(0.00001f));
				BOOST_TEST(round_trip.pointer()[i].imag == 0.0f,
					boost::test_tools::tolerance(0.00001f));
			}
			BOOST_TEST(context.scratch_capacity() == capacity);
			return aExecution;
		});
	const float real[]{1, 0, -1, 0};
	const float imag[]{0, -1, 0, 1};
	for (int i = 0; i < 8; ++i) {
		BOOST_TEST(output.pointer()[i].real == real[i % 4], boost::test_tools::tolerance(0.00001f));
		BOOST_TEST(output.pointer()[i].imag == imag[i % 4], boost::test_tools::tolerance(0.00001f));
	}
}
#endif

} // namespace prepared_tests
