#pragma once

#include <iane/algorithms/fft/spectrum.hpp>
#include <iane/algorithms/wavelet/db2.hpp>
#include <iane/algorithms/wavelet/sym4.hpp>

#include "prepared_cases.hpp"

namespace prepared_tests {

inline void check_spectrum_views(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	const coord_t full_extent{5, 3};
	iane::host_image<iane::fft_complex, 2> packed(3, 3);
	for (int i = 0; i < 9; ++i) {
		packed.pointer()[i] = {static_cast<float>(i), static_cast<float>(i + 1)};
	}
	iane::host_image<iane::fft_complex, 2> centered(full_extent);
	iane::run_on(iane::run_options{.backend = aBackend}, packed.const_view(), centered.view(),
		[full_extent](auto aSource, auto aDestination, auto aExecution) {
			const auto full = iane::hermitian_spectrum_view(aSource, full_extent);
			return iane::copy(iane::fftshift_view(full), aDestination, aExecution);
		});
	// The left side takes both its x and y coordinates from the conjugate partner.
	const int expected[]{5, 4, 6, 7, 8, 2, 1, 0, 1, 2, 8, 7, 3, 4, 5};
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 5; ++x) {
			const int index = expected[5 * y + x];
			const auto pixel = centered.const_view()[coord_t{x, y}];
			BOOST_TEST(pixel.real == index);
			BOOST_TEST(pixel.imag == (x < 2 ? -1 : 1) * (index + 1));
		}
	}

	iane::host_image<iane::fft_complex, 2> even(4, 2);
	iane::host_image<float, 2> display(4, 2);
	for (int i = 0; i < 8; ++i) {
		even.pointer()[i] = {3.0f * i, 4.0f * i};
	}
	iane::run_on(iane::run_options{.backend = aBackend}, even.const_view(), display.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto mapped = iane::log_magnitude_view(iane::fftshift_view(aSource));
			const auto squared = iane::pow(iane::exp(mapped), 2.0f);
			return iane::copy(iane::log(iane::sqrt(squared)), aDestination, aExecution);
		});
	const int order[]{6, 7, 4, 5, 2, 3, 0, 1};
	for (int i = 0; i < 8; ++i) {
		BOOST_TEST(display.pointer()[i] == std::log1p(5.0f * order[i]),
			boost::test_tools::tolerance(0.00001f));
	}
	BOOST_CHECK_THROW(iane::hermitian_spectrum_view(packed.const_view(), (coord_t{6, 3})),
		std::invalid_argument);
	const auto even_full = iane::hermitian_spectrum_view(packed.const_view(), coord_t{4, 3});
	const coord_t nyquist{2, 1};
	BOOST_TEST(even_full[nyquist].imag == 6.0f);
}

template<class TForward, class TInverse>
void check_wavelet(iane::backend aBackend, TForward aForward, TInverse aInverse,
	int aWrapIndex, float aFirst, float aLast, float aWrappedTap)
{
	using coord_t = iane::math::vec2i;
	const coord_t size{10, 8};
	iane::host_image<float, 2> input(12, 8);
	iane::host_image<float, 2> output(size);
	iane::for_each(input.view(), [](float& aPixel) { aPixel = 0.0f; });
	input.pointer()[1] = 1.0f;
	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[=](auto aSource, auto aDestination, auto aExecution) {
			const auto policy = static_cast<iane::view_policy<decltype(aDestination)>>(aExecution);
			auto exec = iane::make_execution_context(policy);
			const auto crop = aSource.subview(coord_t{1, 0}, size);
			auto forward = exec.prepare(aForward, crop, aDestination);
			auto inverse = exec.prepare(aInverse, aDestination, aDestination);
			exec.run(forward, crop, aDestination);
			iane::host_image<float, 2> coefficients(size);
			iane::copy(aDestination, coefficients.view(), policy).wait();
			const auto values = coefficients.pointer();
			BOOST_TEST(values[0] == aFirst * aFirst, boost::test_tools::tolerance(0.00001f));
			BOOST_TEST(values[5] == aFirst * aLast, boost::test_tools::tolerance(0.00001f));
			BOOST_TEST(values[aWrapIndex] == aFirst * aWrappedTap, boost::test_tools::tolerance(0.00001f));
			float energy = 0;
			for (int i = 0; i < 80; ++i) {
				energy += values[i] * values[i];
			}
			BOOST_TEST(energy == 1.0f, boost::test_tools::tolerance(0.00001f));
			const auto result = exec.run(inverse, aDestination, aDestination);
			result.wait();
			return result;
		});
	BOOST_CHECK_SMALL(output.pointer()[0] - 1.0f, 0.00001f);
	for (int i = 1; i < 80; ++i) {
		BOOST_CHECK_SMALL(output.pointer()[i], 0.00001f);
	}
}

inline void check_additional_wavelets(iane::backend aBackend)
{
	check_wavelet(aBackend, iane::db2_forward, iane::db2_inverse,
		4, 0.482962913f, -0.129409523f, 0.224143868f);
	check_wavelet(aBackend, iane::sym4_forward, iane::sym4_inverse,
		2, 0.032223101f, -0.075765715f, -0.029635528f);
}

} // namespace prepared_tests
