#pragma once

#include <iane/algorithms/fft/spectrum.hpp>
#include <iane/algorithms/wavelet/db2.hpp>
#include <iane/algorithms/wavelet/multilevel.hpp>
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

inline void check_multilevel_wavelet(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	const coord_t size{8, 4};
	iane::host_image<float, 2> input(size);
	iane::host_image<float, 2> output(size);
	iane::for_each_position(input.view(), [](float& aPixel, coord_t aPosition) {
		aPixel = static_cast<float>(aPosition[0] + 10 * aPosition[1]);
	});

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[](auto aSource, auto aDestination, auto aExecution) {
			const auto policy = static_cast<iane::view_policy<decltype(aDestination)>>(aExecution);
			auto exec = iane::make_execution_context(policy);
			auto plan = exec.prepare(iane::multilevel_wavelet_forward(iane::wavelet_forward, 2),
				aSource, aDestination);
			BOOST_TEST(exec.scratch_capacity() == 32 * sizeof(float));
			iane::copy(aSource, aDestination, policy);
			return exec.run(plan, aDestination, aDestination);
		});

	auto tree = iane::make_wavelet_pyramid_view(output.const_view(), 2);
	const auto hl = tree.details(iane::wavelet_band::hl);
	const auto lh = tree.details(iane::wavelet_band::lh);
	const auto hh = tree.details(iane::wavelet_band::hh);
	const coord_t zero{};
	const coord_t second{1, 0};
	BOOST_TEST(tree.level_count() == 2u);
	BOOST_TEST(hl.level(0).dimensions()[0] == 4);
	BOOST_TEST(hl.level(0).dimensions()[1] == 2);
	BOOST_TEST(hl.level(1).dimensions()[0] == 2);
	BOOST_TEST(hl.level(1).dimensions()[1] == 1);
	BOOST_TEST(hl.level(0)[second] == -1.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_TEST(lh.level(0)[second] == -10.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_CHECK_SMALL(hh.level(0)[second], 0.00001f);
	BOOST_TEST(hl.level(1)[zero] == -4.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_TEST(lh.level(1)[zero] == -40.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_CHECK_SMALL(hh.level(1)[zero], 0.00001f);
	BOOST_TEST(tree.approximation()[zero] == 66.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_TEST(tree.approximation()[second] == 82.0f, boost::test_tools::tolerance(0.00001f));
	BOOST_CHECK_THROW(iane::make_wavelet_pyramid_view(output.const_view(), 3), std::invalid_argument);
}

template<class TForward, class TInverse>
void check_multilevel_round_trip(iane::backend aBackend, TForward aForward, TInverse aInverse)
{
	using coord_t = iane::math::vec2i;
	const coord_t size{16, 8};
	const coord_t left_origin{1, 1};
	const coord_t right_origin{18, 1};
	iane::host_image<float, 2> input(size);
	iane::host_image<float, 2> output(35, 10);
	for (int i = 0; i < 128; ++i) {
		input.pointer()[i] = static_cast<float>((i * 17) % 31 - 13);
	}

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[=](auto aSource, auto aDestination, auto aExecution) {
			const auto policy = static_cast<iane::view_policy<decltype(aDestination)>>(aExecution);
			auto exec = iane::make_execution_context(policy);
			auto coefficients = aDestination.subview(left_origin, size);
			auto restored = aDestination.subview(right_origin, size);
			auto tree = iane::make_wavelet_pyramid_view(coefficients, 3);
			auto analysis = exec.prepare(iane::multilevel_wavelet_forward(aForward, tree.level_count()),
				aSource, coefficients);
			auto synthesis = exec.prepare(iane::multilevel_wavelet_inverse(aInverse, tree.level_count()),
				tree.coefficients(), restored);
			BOOST_TEST(exec.scratch_capacity() == 128 * sizeof(float));

			iane::copy(iane::constant_view(aDestination, -100.0f), aDestination, policy);
			exec.run(analysis, aSource, coefficients);
			exec.run(synthesis, tree.coefficients(), restored);
			return exec.run(synthesis, tree.coefficients(), coefficients);
		});

	const auto in_place = output.const_view().subview(left_origin, size);
	const auto out_of_place = output.const_view().subview(right_origin, size);
	for (int y = 0; y < size[1]; ++y) {
		for (int x = 0; x < size[0]; ++x) {
			const coord_t position{x, y};
			const auto expected = input.const_view()[position];
			BOOST_CHECK_SMALL(in_place[position] - expected, 0.00003f);
			BOOST_CHECK_SMALL(out_of_place[position] - expected, 0.00003f);
		}
	}
	const coord_t outside{17, 1};
	BOOST_TEST(output.const_view()[outside] == -100.0f);
}

inline void check_multilevel_reconstruction(iane::backend aBackend)
{
	check_multilevel_round_trip(aBackend, iane::wavelet_forward, iane::wavelet_inverse);
	check_multilevel_round_trip(aBackend, iane::db2_forward, iane::db2_inverse);
	check_multilevel_round_trip(aBackend, iane::sym4_forward, iane::sym4_inverse);
}

} // namespace prepared_tests
