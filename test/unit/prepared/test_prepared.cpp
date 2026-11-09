#define BOOST_TEST_MODULE iane_prepared_cpu
#include <boost/test/included/unit_test.hpp>

#include "spectral_cases.hpp"

#if defined(IANE_TEST_FFT)
#include "real_fft_cases.hpp"
#endif
#include "composition_cases.hpp"

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

BOOST_AUTO_TEST_CASE(scan_reduce_and_integral_image) { prepared_tests::check_sequences(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(empty_axis_reduction_and_shape_validation)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<std::uint64_t, 1> output(5);
	const auto source = iane::host::make_const_view<std::uint64_t, 2>(nullptr, coord_t{0, 5});
	const auto algorithm = iane::reduce_along<0>(prepared_tests::sequence_plus{}, std::uint64_t{19});
	auto exec = iane::make_execution_context(iane::cpu_policy{.threads = 4});
	auto reduction = exec.prepare(algorithm, source, output.view());
	exec.run(reduction, source, output.view()).wait();
	for (int i = 0; i < 5; ++i) {
		BOOST_TEST(output.view()[i] == 19u);
	}
	BOOST_TEST(exec.scratch_capacity() == 0u);
	iane::host_image<std::uint64_t, 1> wrong_output(4);
	BOOST_CHECK_THROW(exec.prepare(algorithm, source, wrong_output.view()), std::invalid_argument);
	BOOST_CHECK_THROW(exec.run(reduction, source, wrong_output.view()), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(histogram_equalization_and_stable_compaction)
{
	prepared_tests::check_histogram_and_compaction(iane::backend::cpu);
}

BOOST_AUTO_TEST_CASE(pyramid_with_two_construction_rules_and_borrowed_levels)
{
	using image_t = iane::host_image<float, 2>;
	using coord_t = iane::math::vec2i;
	const coord_t extents[]{coord_t{5, 2}, coord_t{3, 2}, coord_t{2, 2}};
	const auto build = [&](auto aCombine) {
		iane::image_pyramid<image_t> pyramid(extents);
		auto levels = pyramid.view();
		iane::for_each_position(levels.level(0), [](float& aPixel, coord_t aPosition) {
			aPixel = static_cast<float>(aPosition[0] + 10 * aPosition[1]);
		});
		for (std::size_t i = 1; i < levels.level_count(); ++i) {
			const auto source = levels.level(i - 1);
			const auto destination = levels.level(i);
			const int last = source.dimensions()[0] - 1;
			auto samples = iane::generate_view(destination, [=](coord_t aPosition) {
				const int x = 2 * aPosition[0];
				const int y = aPosition[1];
				return aCombine(source[coord_t{std::max(0, x - 1), y}], source[coord_t{x, y}],
					source[coord_t{std::min(last, x + 1), y}]);
			});
			iane::copy(samples, destination);
		}
		return pyramid;
	};
	auto gaussian = build([](float aLeft, float aCenter, float aRight) {
		return (aLeft + 2 * aCenter + aRight) / 4;
	});
	auto maximum = build([](float aLeft, float aCenter, float aRight) {
		return std::max({aLeft, aCenter, aRight});
	});
	const auto smooth = gaussian.const_view();
	const auto peak = maximum.const_view();
	const coord_t left{0, 0};
	const coord_t right{1, 1};
	BOOST_TEST(smooth.level_count() == 3u);
	BOOST_TEST(smooth.level(2)[left] == 0.6875f);
	BOOST_TEST(smooth.level(2)[right] == 13.3125f);
	BOOST_TEST(peak.level(2)[left] == 3.0f);
	BOOST_TEST(peak.level(2)[right] == 14.0f);

	const auto cropped = smooth.level(0).subview(coord_t{1, 0}, coord_t{4, 2});
	const image_t::const_view_t table[]{cropped, peak.level(2)};
	const iane::pyramid_view<image_t::const_view_t> borrowed(table);
	BOOST_TEST(borrowed.level(0)[left] == 1.0f);
	BOOST_TEST(borrowed.level(1)[right] == 14.0f);
	BOOST_CHECK_THROW(borrowed.level(2), std::out_of_range);
}

BOOST_AUTO_TEST_CASE(histogram_ignores_invalid_indices_and_constant_cdf_uses_identity)
{
	const int values[]{-1, 1, 3, 1};
	auto source = iane::host::make_const_view<int, 1>(values, 4);
	iane::host_image<unsigned int, 1> bins(3);
	iane::host_image<unsigned int, 1> cdf(3);
	iane::host_image<float, 1> lut(3);
	iane::histogram(source, bins.view(), prepared_tests::pixel_bin{});
	BOOST_TEST(bins.view()[0] == 0u);
	BOOST_TEST(bins.view()[1] == 2u);
	BOOST_TEST(bins.view()[2] == 0u);
	auto exec = iane::make_execution_context();
	auto scan = exec.prepare(iane::inclusive_scan(prepared_tests::add_count{}, 0u), bins.const_view(), cdf.view());
	exec.run(scan, bins.const_view(), cdf.view());
	iane::equalization_lut(cdf.const_view(), lut.view());
	BOOST_TEST(lut.view()[0] == 0.0f);
	BOOST_TEST(lut.view()[1] == 0.5f);
	BOOST_TEST(lut.view()[2] == 1.0f);
}

BOOST_AUTO_TEST_CASE(spectral_display_views) { prepared_tests::check_spectrum_views(iane::backend::cpu); }

BOOST_AUTO_TEST_CASE(additional_wavelets) { prepared_tests::check_additional_wavelets(iane::backend::cpu); }

#if defined(IANE_TEST_FFT)
BOOST_AUTO_TEST_CASE(real_fft_odd_and_even_round_trips)
{
	using coord_t = iane::math::vec2i;
	prepared_tests::check_real_fft(iane::backend::cpu, coord_t{4, 3});
	prepared_tests::check_real_fft(iane::backend::cpu, coord_t{5, 3});
	prepared_tests::check_real_fft(iane::backend::cpu, coord_t{6, 4});
}

BOOST_AUTO_TEST_CASE(real_fft_rejects_invalid_layout_and_reuse)
{
	using coord_t = iane::math::vec2i;
	iane::host_image<float, 2> odd(5, 3);
	iane::host_image<float, 2> even(4, 3);
	iane::host_image<float, 2> padded(6, 3);
	iane::host_image<iane::fft_complex, 2> packed(3, 3);
	iane::host_image<iane::fft_complex, 2> wrong(2, 3);
	auto exec = iane::make_execution_context();
	auto inverse = exec.prepare(iane::irfft, packed.const_view(), odd.view());
	BOOST_CHECK_THROW(exec.run(inverse, packed.const_view(), even.view()), std::invalid_argument);
	BOOST_CHECK_THROW(exec.prepare(iane::rfft, odd.const_view(), wrong.view()), std::invalid_argument);
	const auto crop = padded.const_view().subview(coord_t{0, 0}, odd.dimensions());
	BOOST_CHECK_THROW(exec.prepare(iane::rfft, crop, packed.view()), std::invalid_argument);
}
#endif

BOOST_AUTO_TEST_CASE(multilevel_wavelet_pyramid)
{
	prepared_tests::check_multilevel_wavelet(iane::backend::cpu);
}

BOOST_AUTO_TEST_CASE(multilevel_wavelet_validates_levels_and_reused_extent)
{
	iane::host_image<float, 2> input(8, 4);
	iane::host_image<float, 2> output(8, 4);
	iane::host_image<float, 2> other(4, 4);
	auto exec = iane::make_execution_context();
	const auto valid = iane::multilevel_wavelet_forward(iane::db2_forward, 2);
	auto plan = exec.prepare(valid, input.const_view(), output.view());
	BOOST_CHECK_THROW(exec.run(plan, other.const_view(), other.view()), std::invalid_argument);
	BOOST_CHECK_THROW(exec.prepare(iane::multilevel_wavelet_forward(iane::db2_forward, 0),
		input.const_view(), output.view()), std::invalid_argument);
	BOOST_CHECK_THROW(exec.prepare(iane::multilevel_wavelet_forward(iane::db2_forward, 3),
		input.const_view(), output.view()), std::invalid_argument);
}
