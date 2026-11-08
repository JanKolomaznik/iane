#pragma once

#include "sequence_cases.hpp"

#include <iane/algorithms/compact.hpp>
#include <iane/algorithms/histogram.hpp>
#include <iane/image/pyramid.hpp>

namespace prepared_tests {

// Use pixel values as histogram and lookup indices.
struct pixel_bin {
	IANE_DECL_HYBRID int operator()(int aPixel) const { return aPixel; }
};

// Select the pixels whose logical positions should be compacted.
struct selected_pixel {
	IANE_DECL_HYBRID bool operator()(int aPixel) const { return aPixel == 2; }
};

// Add cumulative histogram counts.
struct add_count {
	IANE_DECL_HYBRID unsigned int operator()(unsigned int aLeft, unsigned int aRight) const
	{
		return aLeft + aRight;
	}
};

inline void check_histogram_and_compaction(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	constexpr int width = 259;
	constexpr int capacity = 153;
	const coord_t size{width, 2};
	iane::host_image<int, 2> source(size);
	iane::host_image<float, 2> output(size);
	unsigned int expected[4]{};
	std::vector<coord_t> selected;
	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < width; ++x) {
			const int value = 1 + (x + y) % 3;
			const coord_t position{x, y};
			source.view()[position] = value;
			++expected[value];
			if (value == 2) {
				selected.push_back(position);
			}
		}
	}
	iane::run_on(iane::run_options{.backend = aBackend}, source.const_view(), output.view(),
		[&](auto aSource, auto aDestination, auto aExecution) {
			using view_t = decltype(aDestination);
			auto policy = static_cast<iane::view_policy<view_t>>(aExecution);
			if constexpr (iane::concepts::HostView<view_t>) {
				policy = {.threads = 4, .grain_size = 29};
			}
			using level_t = sequence_image<unsigned int, 1, view_t>;
			const int extents[]{4, 4};
			iane::image_pyramid<level_t> tables(extents);
			const auto bins = tables.view().level(0);
			const auto cdf = tables.view().level(1);
			sequence_image<float, 1, view_t> lut(4);
			sequence_image<coord_t, 1, view_t> coordinates(capacity);
			sequence_image<std::size_t, 1, view_t> count(1);
			iane::host_image<coord_t, 1> actual(capacity);
			iane::host_image<std::size_t, 1> total(1);
			iane::host_image<unsigned int, 1> actual_bins(4);
			auto exec = iane::make_execution_context(policy);
			auto scan = exec.prepare(iane::inclusive_scan(add_count{}, 0u), bins, cdf);
			auto destination = iane::compact_output{coordinates.view(), count.view()};
			auto compact = exec.prepare(iane::compact(selected_pixel{}), aSource, destination);

			iane::histogram(aSource, bins, pixel_bin{}, policy);
			iane::histogram(aSource, bins, pixel_bin{}, policy);
			iane::copy(bins, actual_bins.view(), policy).wait();
			for (int i = 0; i < 4; ++i) {
				BOOST_TEST(actual_bins.view()[i] == expected[i]);
			}
			exec.run(scan, bins, cdf);
			iane::equalization_lut(cdf, lut.view(), policy);
			iane::copy(iane::lookup_view(aSource, lut.const_view(), pixel_bin{}), aDestination, policy);

			exec.run(compact, aSource, destination);
			iane::copy(coordinates.const_view(), actual.view(), policy);
			iane::copy(count.const_view(), total.view(), policy).wait();
			BOOST_TEST(total.view()[0] == selected.size());
			for (int i = 0; i < capacity; ++i) {
				BOOST_TEST(actual.view()[i][0] == selected[i][0]);
				BOOST_TEST(actual.view()[i][1] == selected[i][1]);
			}
			destination.coordinates = coordinates.view().subview(0, 0);
			exec.run(compact, aSource, destination);
			iane::copy(count.const_view(), total.view(), policy).wait();
			BOOST_TEST(total.view()[0] == selected.size());
			return aExecution;
		});

	const float middle = static_cast<float>(expected[2]) / (expected[2] + expected[3]);
	for (int y = 0; y < 2; ++y) {
		for (int x = 0; x < width; ++x) {
			const float values[]{0.0f, middle, 1.0f};
			const coord_t position{x, y};
			BOOST_TEST(output.view()[position] == values[source.view()[position] - 1],
				boost::test_tools::tolerance(0.000001f));
		}
	}
}

} // namespace prepared_tests
