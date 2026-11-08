#pragma once

#include <iane/algorithms/detail/sequence.hpp>

namespace iane {

/**
 * @brief Destination views for stable coordinate compaction.
 *
 * The count is a writable 1D std::size_t view of extent one. It reports all
 * matches, including those beyond the coordinate buffer's capacity.
 */
template<class TCoordinates, class TCount>
struct compact_output {
	TCoordinates coordinates;
	TCount count;
};

namespace detail {

enum class compact_stage { count, offsets, write };

// Count matches per chunk, turn counts into offsets, then emit stable coordinates.
template<compact_stage tStage, class TSource, class TOutput, class TPredicate>
struct compact_task {
	TSource source;
	TOutput output;
	TPredicate predicate;
	typename TSource::extents_t extent;
	std::size_t size;
	std::size_t grain;
	std::size_t chunks;
	std::size_t capacity;
	std::size_t* offsets;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void operator()(std::size_t aChunk)
	{
		if constexpr (tStage == compact_stage::offsets) {
			std::size_t total = 0;
			for (std::size_t i = 0; i < chunks; ++i) {
				const auto count = offsets[i];
				offsets[i] = total;
				total += count;
			}
			output.count[0] = total;
		} else {
			const auto begin = aChunk * grain;
			const auto remaining = size - begin;
			const auto end = begin + (remaining < grain ? remaining : grain);
			std::size_t index = 0;
			if constexpr (tStage == compact_stage::write) {
				index = offsets[aChunk];
			}
			for (auto i = begin; i < end; ++i) {
				if constexpr (tStage == compact_stage::write) {
					if (index >= capacity) {
						break;
					}
				}
				const auto position = math::detail::coordinate_at<TSource::dimension, typename TSource::coord_t>(
					i, extent);
				if (predicate(source[position])) {
					if constexpr (tStage == compact_stage::write) {
						output.coordinates[index] = position;
					}
					++index;
				}
			}
			if constexpr (tStage == compact_stage::count) {
				offsets[aChunk] = index;
			}
		}
	}
};

// Store the prepared source shape and chunk partition for reusable compaction.
template<class TSource, class TOutput, class TPredicate, class TPolicy>
struct compact_state {
	typename TSource::extents_t extent;
	std::size_t size;
	std::size_t grain;
	std::size_t chunks;
	TPredicate predicate;

	std::size_t scratch_bytes() const
	{
		if (chunks > std::numeric_limits<std::size_t>::max() / sizeof(std::size_t)) {
			throw std::overflow_error("Compaction scratch size overflows");
		}
		return chunks * sizeof(std::size_t);
	}

	auto run(TSource aSource, TOutput aOutput, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		const auto actual = aSource.dimensions();
		for (int i = 0; i < TSource::dimension; ++i) {
			if (math::detail::axis(actual, i) != math::detail::axis(extent, i)) {
				throw std::invalid_argument("Compaction source must match the prepared shape");
			}
		}
		const auto capacity = validate_output(aOutput);
		auto* offsets = static_cast<std::size_t*>(aScratch.bytes(scratch_bytes(), alignof(std::size_t)));
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			aPolicy.grain_size = 1;
		}
		visit_coordinates(sequence_domain<TPolicy>{chunks},
			compact_task<compact_stage::count, TSource, TOutput, TPredicate>{
				aSource, aOutput, predicate, extent, size, grain, chunks, capacity, offsets}, aPolicy);
		visit_coordinates(sequence_domain<TPolicy>{1},
			compact_task<compact_stage::offsets, TSource, TOutput, TPredicate>{
				aSource, aOutput, predicate, extent, size, grain, chunks, capacity, offsets}, aPolicy);
		return visit_coordinates(sequence_domain<TPolicy>{chunks},
			compact_task<compact_stage::write, TSource, TOutput, TPredicate>{
				aSource, aOutput, predicate, extent, size, grain, chunks, capacity, offsets}, aPolicy);
	}

	static std::size_t validate_output(TOutput aOutput)
	{
		if (math::detail::element_count<1>(aOutput.count.dimensions()) != 1) {
			throw std::invalid_argument("Compaction count requires an extent-one view");
		}
		return math::detail::element_count<1>(aOutput.coordinates.dimensions());
	}
};

// Validate the views and plan scratch without retaining borrowed buffers.
template<class TPredicate>
struct compact_algorithm {
	TPredicate predicate;

	template<concepts::ReadableView TSource, concepts::ViewDomain TCoordinates,
		concepts::ViewDomain TCount, class TPolicy>
	requires (TCoordinates::dimension == 1 && TCount::dimension == 1) &&
		std::same_as<typename TCoordinates::value_type, typename TSource::coord_t> &&
		std::same_as<typename TCount::value_type, std::size_t> &&
		concepts::WritableView<TCoordinates> && concepts::WritableView<TCount> &&
		concepts::SameViewBackend<TSource, TCoordinates> && concepts::SameViewBackend<TSource, TCount> &&
		std::same_as<TPolicy, view_policy<TSource>> && std::same_as<TPolicy, view_policy<TCoordinates>> &&
		std::same_as<TPolicy, view_policy<TCount>>
	auto prepare(TSource aSource, compact_output<TCoordinates, TCount> aOutput, TPolicy aPolicy) const
	{
		using state_t = compact_state<TSource, decltype(aOutput), TPredicate, TPolicy>;
		state_t::validate_output(aOutput);
		const typename TSource::extents_t extent = aSource.dimensions();
		const auto size = math::detail::element_count<TSource::dimension>(extent);
		std::size_t grain = 128;
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			if (aPolicy.threads == 0 || aPolicy.grain_size == 0) {
				throw std::invalid_argument("CPU policy requires positive threads and grain size");
			}
			grain = aPolicy.grain_size;
		}
		return state_t{extent, size, grain, size == 0 ? 0 : 1 + (size - 1) / grain, predicate};
	}
};

} // namespace detail

/**
 * @brief Configure stable selection of source coordinates for execution_context::prepare.
 * @param aPredicate Pure pixel predicate, callable on the selected backend.
 * @return Prepared algorithm factory for a source view and compact_output.
 *
 * The first matches that fit are written in axis-zero-fastest order. The
 * full count is written even when capacity is zero. Unused coordinates
 * stay untouched. Output capacity may change on reuse. All storage must
 * be disjoint and stay alive through completion.
 */
template<class TPredicate>
auto compact(TPredicate aPredicate)
{
	return detail::compact_algorithm<TPredicate>{aPredicate};
}

} // namespace iane
