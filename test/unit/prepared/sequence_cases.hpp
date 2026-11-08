#pragma once

#include <iane/algorithms/reduce.hpp>
#include <iane/algorithms/scan.hpp>
#include <iane/execution/run_on.hpp>

#include <boost/test/unit_test.hpp>
#include <cstdint>

namespace prepared_tests {

// Allocate the intermediate and reductions on the backend selected by run_on.
template<class TElement, int tDim, class TView>
using sequence_image =
#if defined(__CUDACC__)
	std::conditional_t<iane::concepts::DeviceView<TView>, iane::device_image<TElement, tDim>,
		iane::host_image<TElement, tDim>>;
#else
	iane::host_image<TElement, tDim>;
#endif

// Widen the accumulation independently of the input pixel type.
struct sequence_plus {
	IANE_DECL_HYBRID std::uint64_t operator()(std::uint64_t aLeft, std::uint64_t aRight) const
	{
		return aLeft + aRight;
	}
};

inline void check_sequences(iane::backend aBackend)
{
	using coord_t = iane::math::vec2i;
	constexpr int width = 263;
	constexpr int height = 5;
	const coord_t size{width, height};
	iane::host_image<std::uint8_t, 2> input(size);
	iane::host_image<std::uint64_t, 2> output(size);
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			input.view()[coord_t{x, y}] = static_cast<std::uint8_t>(x % 7 + y);
		}
	}

	iane::run_on(iane::run_options{.backend = aBackend}, input.const_view(), output.view(),
		[size](auto aSource, auto aDestination, auto aExecution) {
			using view_t = decltype(aDestination);
			auto policy = static_cast<iane::view_policy<view_t>>(aExecution);
			if constexpr (iane::concepts::HostView<view_t>) {
				policy = {.threads = 8, .grain_size = 31};
			}
			sequence_image<std::uint64_t, 2, view_t> intermediate(coord_t{width + 2, height});
			auto rows = intermediate.view().subview(coord_t{1, 0}, size);
			sequence_image<std::uint64_t, 1, view_t> scalar(1);
			sequence_image<std::uint64_t, 1, view_t> row_totals(height);
			auto exec = iane::make_execution_context(policy);
			constexpr std::uint64_t seed = 11;
			constexpr sequence_plus plus{};
			auto row_scan = exec.prepare(iane::inclusive_scan_along<0>(plus, std::uint64_t{}), aSource, rows);
			auto column_scan = exec.prepare(iane::inclusive_scan_along<1>(plus, std::uint64_t{}), rows, aDestination);
			auto flat_scan = exec.prepare(iane::exclusive_scan(plus, seed), aSource, aDestination);
			auto full_reduce = exec.prepare(iane::reduce(plus, seed), aSource, scalar.view());
			auto row_reduce = exec.prepare(iane::reduce_along<0>(plus, seed), aSource, row_totals.view());
			auto exclusive_columns = exec.prepare(iane::exclusive_scan_along<1>(plus, seed), aSource, aDestination);
			auto inclusive_flat = exec.prepare(iane::inclusive_scan(plus, seed), aSource, aDestination);

			iane::host_image<std::uint64_t, 1> total(1);
			iane::host_image<std::uint64_t, 1> totals(height);
			iane::host_image<std::uint64_t, 2> prefixes(size);
			exec.run(full_reduce, aSource, scalar.view());
			exec.run(row_reduce, aSource, row_totals.view());
			iane::copy(scalar.const_view(), total.view(), policy).wait();
			iane::copy(row_totals.const_view(), totals.view(), policy).wait();
			for (bool inclusive : {false, true}) {
				if (inclusive) {
					exec.run(inclusive_flat, aSource, aDestination);
				} else {
					exec.run(flat_scan, aSource, aDestination);
				}
				iane::copy(aDestination, prefixes.view(), policy).wait();
				auto prefix = seed;
				for (int y = 0; y < height; ++y) {
					auto row = seed;
					for (int x = 0; x < width; ++x) {
						const auto pixel = static_cast<std::uint64_t>(x % 7 + y);
						BOOST_TEST((prefixes.view()[coord_t{x, y}]) == prefix + (inclusive ? pixel : 0));
						prefix += pixel;
						row += pixel;
					}
					BOOST_TEST(totals.view()[y] == row);
				}
				BOOST_TEST(total.view()[0] == prefix);
			}
			exec.run(exclusive_columns, aSource, aDestination);
			iane::copy(aDestination, prefixes.view(), policy).wait();
			for (int x = 0; x < width; ++x) {
				auto prefix = seed;
				for (int y = 0; y < height; ++y) {
					BOOST_TEST((prefixes.view()[coord_t{x, y}]) == prefix);
					prefix += x % 7 + y;
				}
			}

			exec.run(row_scan, aSource, rows);
			const auto result = exec.run(column_scan, rows, aDestination);
			result.wait();
			return result;
		});

	for (int y = 0; y < height; ++y) {
		std::uint64_t rectangle = 0;
		for (int x = 0; x < width; ++x) {
			for (int row = 0; row <= y; ++row) {
				rectangle += x % 7 + row;
			}
			BOOST_TEST((output.const_view()[coord_t{x, y}]) == rectangle);
		}
	}
}

} // namespace prepared_tests
