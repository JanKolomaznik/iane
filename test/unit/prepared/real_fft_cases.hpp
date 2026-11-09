#pragma once

#include "prepared_cases.hpp"

#include <iane/algorithms/fft/rfft.hpp>
#include <iane/algorithms/fft/spectrum.hpp>

#include <numbers>

namespace prepared_tests {

inline void check_real_fft(iane::backend aBackend, iane::math::vec2i aSize)
{
	using coord_t = iane::math::vec2i;
	const auto packed_size = iane::rfft_extent(aSize);
	iane::host_image<float, 2> input(aSize);
	iane::host_image<float, 2> output(aSize);
	iane::host_image<iane::fft_complex, 2> before(packed_size);
	iane::host_image<iane::fft_complex, 2> after(packed_size);
	iane::host_image<float, 2> display(aSize);
	const int count = aSize[0] * aSize[1];
	for (int i = 0; i < count; ++i) {
		input.pointer()[i] = static_cast<float>((i * 7) % 11 - 4);
	}

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[&](auto aSource, auto aDestination, auto aExecution) {
			using view_t = decltype(aDestination);
			const auto policy = static_cast<iane::view_policy<view_t>>(aExecution);
			backend_image<iane::fft_complex, view_t> packed(packed_size);
			backend_image<float, view_t> mapped(aSize);
			auto exec = iane::make_execution_context(policy);
			auto forward = exec.prepare(iane::rfft, aSource, packed.view());
			auto inverse = exec.prepare(iane::irfft, packed.const_view(), aDestination);
			exec.run(forward, aSource, packed.view());
			iane::copy(packed.const_view(), before.view(), policy);
			exec.run(inverse, packed.const_view(), aDestination);
			iane::copy(packed.const_view(), after.view(), policy);
			const auto full = iane::hermitian_spectrum_view(packed.const_view(), aSize);
			iane::copy(iane::log_magnitude_view(iane::fftshift_view(full)), mapped.view(), policy);
			iane::copy(mapped.const_view(), display.view(), policy).wait();
			return aExecution;
		});

	for (int i = 0; i < count; ++i) {
		const float expected = count * static_cast<float>((i * 7) % 11 - 4);
		BOOST_CHECK_SMALL(output.pointer()[i] - expected, 0.0002f);
		BOOST_TEST(input.pointer()[i] == static_cast<float>((i * 7) % 11 - 4));
	}
	for (int ky = 0; ky < aSize[1]; ++ky) {
		for (int kx = 0; kx < aSize[0]; ++kx) {
			double real = 0;
			double imag = 0;
			for (int y = 0; y < aSize[1]; ++y) {
				for (int x = 0; x < aSize[0]; ++x) {
					const auto phase = -2 * std::numbers::pi *
						(static_cast<double>(kx * x) / aSize[0] + static_cast<double>(ky * y) / aSize[1]);
					const auto pixel = input.const_view()[coord_t{x, y}];
					real += pixel * std::cos(phase);
					imag += pixel * std::sin(phase);
				}
			}
			if (kx < packed_size[0]) {
				const coord_t bin{kx, ky};
				const auto saved = before.const_view()[bin];
				const auto preserved = after.const_view()[bin];
				BOOST_CHECK_SMALL(static_cast<double>(saved.real) - real, 0.00003);
				BOOST_CHECK_SMALL(static_cast<double>(saved.imag) - imag, 0.00003);
				BOOST_TEST(preserved.real == saved.real);
				BOOST_TEST(preserved.imag == saved.imag);
			}
			const coord_t centered{(kx + aSize[0] / 2) % aSize[0], (ky + aSize[1] / 2) % aSize[1]};
			const auto expected = std::log1p(std::hypot(real, imag));
			BOOST_CHECK_SMALL(static_cast<double>(display.const_view()[centered]) - expected, 0.00003);
		}
	}
}

} // namespace prepared_tests
