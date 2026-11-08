#pragma once

#include <iane/algorithms/copy.hpp>
#include <iane/view/procedural.hpp>

#include <atomic>
#include <limits>

namespace iane {
namespace detail {

// Increment the selected bin using the backend's atomic operation.
template<class TSource, class TBins, class TBin>
struct histogram_visit {
	TSource source;
	TBins bins;
	TBin bin;
	std::size_t size;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void operator()(const typename TSource::coord_t& aPosition)
	{
		const auto index = static_cast<std::size_t>(bin(source[aPosition]));
		if (index < size) {
#if defined(__CUDA_ARCH__)
			atomicAdd(&bins[static_cast<int>(index)], 1u);
#else
			std::atomic_ref<unsigned int>(bins[static_cast<int>(index)]).fetch_add(1, std::memory_order_relaxed);
#endif
		}
	}
};

// Read the cumulative counts and derive one normalized equalization entry.
template<class TCdf>
struct equalization_pixel {
	TCdf cdf;
	int size;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID float operator()(int aIndex) const
	{
		const auto total = cdf[size - 1];
		int first = 0;
		int last = size - 1;
		while (first < last) {
			const auto middle = first + (last - first) / 2;
			if (cdf[middle] == 0) {
				first = middle + 1;
			} else {
				last = middle;
			}
		}
		const auto minimum = cdf[first];
		if (total == minimum) {
			return size > 1 ? static_cast<float>(aIndex) / (size - 1) : 0.0f;
		}
		return aIndex < first ? 0.0f : static_cast<float>(cdf[aIndex] - minimum) / (total - minimum);
	}
};

// Evaluate a lookup without materializing the source or the remapped image.
template<class TSource, class TTable, class TBin>
struct lookup_pixel {
	TSource source;
	TTable table;
	TBin bin;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID typename TTable::value_type operator()(const typename TSource::coord_t& aPosition) const
	{
		return table[bin(source[aPosition])];
	}
};

} // namespace detail

/**
 * @brief Replace the counts in a 1D unsigned int histogram.
 * @param aSource Readable input, disjoint from the bins.
 * @param aBins Distinct counters on the source backend, aligned for atomic access.
 * @param aBin Pure integral bin mapper. Out-of-range indices are ignored.
 * @param aPolicy Backend scheduling and stream policy.
 * @return Completion handle. Keep storage alive through completion.
 * @throws std::overflow_error If the source element count exceeds counter capacity.
 */
template<concepts::ReadableView TSource, concepts::ViewDomain TBins, class TBin>
requires concepts::SameViewBackend<TSource, TBins> && (TBins::dimension == 1) &&
	std::same_as<typename TBins::coord_t, int> &&
	std::integral<std::invoke_result_t<TBin&, typename TSource::value_type>> &&
	requires(TBins aBins) { { aBins[0] } -> std::same_as<unsigned int&>; }
auto histogram(TSource aSource, TBins aBins, TBin aBin, view_policy<TBins> aPolicy = {})
{
	const auto count = math::detail::element_count<TSource::dimension>(aSource.dimensions());
	if (count > std::numeric_limits<unsigned int>::max()) {
		throw std::overflow_error("Histogram source exceeds counter capacity");
	}
	const auto size = math::detail::element_count<1>(aBins.dimensions());
	copy(constant_view(aBins, 0u), aBins, aPolicy);
	return detail::visit_coordinates(aSource,
		detail::histogram_visit<TSource, TBins, TBin>{aSource, aBins, aBin, size}, aPolicy);
}

/**
 * @brief Build a normalized float lookup table from a nondecreasing unsigned CDF.
 * @pre Both views have the same extent and backend and disjoint storage.
 * @return Completion handle. Empty and single-bin distributions use an identity ramp.
 */
template<concepts::ReadableView TCdf, concepts::ViewDomain TLut>
requires concepts::TransformViews<TCdf, TLut> && (TCdf::dimension == 1) &&
	std::same_as<typename TCdf::coord_t, int> &&
	std::same_as<typename TCdf::value_type, unsigned int> &&
	std::same_as<typename TLut::value_type, float>
auto equalization_lut(TCdf aCdf, TLut aLut, view_policy<TLut> aPolicy = {})
{
	return copy(generate_view(aCdf, detail::equalization_pixel<TCdf>{aCdf, aCdf.dimensions()}), aLut, aPolicy);
}

/**
 * @brief Borrow a lazy remapping of the source through a 1D table.
 * @pre Mapped indices are in range. Both operands share a backend and
 * their storage remains valid through evaluation and completion.
 */
template<concepts::ReadableView TSource, concepts::ReadableView TTable, class TBin>
requires concepts::SameViewBackend<TSource, TTable> && (TTable::dimension == 1)
auto lookup_view(TSource aSource, TTable aTable, TBin aBin)
{
	return generate_view(aSource, detail::lookup_pixel<TSource, TTable, TBin>{aSource, aTable, aBin});
}

} // namespace iane
