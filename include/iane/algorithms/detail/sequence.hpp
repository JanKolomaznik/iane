#pragma once

#include <iane/algorithms/for_each.hpp>
#include <iane/execution/prepared.hpp>

namespace iane::detail {

enum class sequence_kind { inclusive, exclusive, reduction };

// Tag a linear work domain for the existing CPU and CUDA traversal schedulers.
template<class TPolicy>
struct sequence_domain;

// Provide the host tag for work submitted to CPU workers.
template<>
struct sequence_domain<cpu_policy> {
	using is_host_view_tag = void;
	using coord_t = std::size_t;
	using extents_t = std::size_t;
	using value_type = std::size_t;
	static constexpr int dimension = 1;
	std::size_t count;
	std::size_t dimensions() const { return count; }
};

#if defined(__CUDACC__)
// Provide the device tag for work submitted to the CUDA stream.
template<>
struct sequence_domain<cuda_policy> {
	using is_device_view_tag = void;
	using coord_t = std::size_t;
	using extents_t = std::size_t;
	using value_type = std::size_t;
	static constexpr int dimension = 1;
	std::size_t count;
	std::size_t dimensions() const { return count; }
};
#endif

// Map independent sequences and their elements to logical view coordinates.
template<int tAxis, class TSrc>
struct sequence_layout {
	typename TSrc::extents_t extent;
	std::size_t lines;
	std::size_t length;
	std::size_t chunk_size;
	std::size_t chunks;

	IANE_DECL_HYBRID auto position(std::size_t aLine, std::size_t aIndex) const
	{
		using namespace math::detail;
		if constexpr (tAxis < 0) {
			return coordinate_at<TSrc::dimension, typename TSrc::coord_t>(aIndex, extent);
		} else {
			typename TSrc::coord_t position{};
			for (int axis_index = 0; axis_index < TSrc::dimension; ++axis_index) {
				if (axis_index == tAxis) {
					axis(position, axis_index) = aIndex;
				} else {
					const auto size = static_cast<std::size_t>(axis(extent, axis_index));
					axis(position, axis_index) = aLine % size;
					aLine /= size;
				}
			}
			return position;
		}
	}
};

// Perform the chunk totals, line carries, or final output pass.
template<int tAxis, sequence_kind tKind, class TSrc, class TDst, class TOp, class TValue>
struct sequence_pass {
	sequence_layout<tAxis, TSrc> layout;
	TSrc source;
	TDst destination;
	typename TDst::extents_t output_extent;
	TOp op;
	TValue init;
	TValue* scratch;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void reduce_chunk(std::size_t aTask)
	{
		const auto line = aTask / layout.chunks;
		const auto chunk = aTask % layout.chunks;
		if constexpr (tKind != sequence_kind::reduction) {
			if (chunk + 1 == layout.chunks) {
				return;
			}
		}
		const auto begin = chunk * layout.chunk_size;
		const auto end = begin + chunk_length(begin);
		auto total = static_cast<TValue>(source[layout.position(line, begin)]);
		for (auto i = begin + 1; i < end; ++i) {
			total = op(total, static_cast<TValue>(source[layout.position(line, i)]));
		}
		scratch[aTask] = total;
	}

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void combine(std::size_t aLine)
	{
		auto total = init;
		const auto begin = aLine * layout.chunks;
		const auto partials = layout.chunks - (tKind != sequence_kind::reduction);
		for (std::size_t i = 0; i < partials; ++i) {
			const auto partial = scratch[begin + i];
			if constexpr (tKind != sequence_kind::reduction) {
				scratch[begin + i] = total;
			}
			total = op(total, partial);
		}
		if constexpr (tKind == sequence_kind::reduction) {
			write_reduction(aLine, total);
		} else {
			scratch[begin + partials] = total;
		}
	}

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void write(std::size_t aTask)
	{
		const auto line = aTask / layout.chunks;
		const auto begin = (aTask % layout.chunks) * layout.chunk_size;
		const auto end = begin + chunk_length(begin);
		auto total = scratch ? scratch[aTask] : init;
		for (auto i = begin; i < end; ++i) {
			const auto position = layout.position(line, i);
			if constexpr (tKind == sequence_kind::exclusive) {
				destination[position] = total;
				if (i + 1 == end) {
					break;
				}
			}
			total = op(total, static_cast<TValue>(source[position]));
			if constexpr (tKind == sequence_kind::inclusive) {
				destination[position] = total;
			}
		}
		if constexpr (tKind == sequence_kind::reduction) {
			write_reduction(line, total);
		}
	}

	IANE_DECL_HYBRID std::size_t chunk_length(std::size_t aBegin) const
	{
		const auto remaining = layout.length - aBegin;
		return remaining < layout.chunk_size ? remaining : layout.chunk_size;
	}

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void write_reduction(std::size_t aLine, TValue aTotal)
	{
		if constexpr (tAxis < 0 || TSrc::dimension == 1) {
			destination[0] = aTotal;
		} else {
			destination[math::detail::coordinate_at<TDst::dimension, typename TDst::coord_t>(
				aLine, output_extent)] = aTotal;
		}
	}
};

enum class sequence_stage { totals, carries, output };

// Adapt each pass to a traversal callback shared by the CPU and CUDA schedulers.
template<sequence_stage tStage, class TPass>
struct sequence_task {
	TPass pass;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID void operator()(std::size_t aIndex)
	{
		if constexpr (tStage == sequence_stage::totals) {
			pass.reduce_chunk(aIndex);
		} else if constexpr (tStage == sequence_stage::carries) {
			pass.combine(aIndex);
		} else {
			pass.write(aIndex);
		}
	}
};

// Retain shape and chunking independently of the prototype views' storage.
template<int tAxis, sequence_kind tKind, class TSrc, class TDst, class TOp, class TValue, class TPolicy>
struct sequence_state {
	sequence_layout<tAxis, TSrc> layout;
	typename TDst::extents_t output_extent;
	TOp op;
	TValue init;

	std::size_t scratch_bytes() const
	{
		if (layout.chunks == 1) {
			return 0;
		}
		const auto tasks = layout.lines * layout.chunks;
		if (tasks > std::numeric_limits<std::size_t>::max() / sizeof(TValue)) {
			throw std::overflow_error("Sequence scratch size overflows");
		}
		return tasks * sizeof(TValue);
	}

	auto run(TSrc aSource, TDst aDestination, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		check_extent<TSrc>(aSource.dimensions(), layout.extent);
		check_extent<TDst>(aDestination.dimensions(), output_extent);
		using pass_t = sequence_pass<tAxis, tKind, TSrc, TDst, TOp, TValue>;
		auto* scratch = static_cast<TValue*>(aScratch.bytes(scratch_bytes(), alignof(TValue)));
		const pass_t pass{layout, aSource, aDestination, output_extent, op, init, scratch};
		const auto tasks = layout.lines * layout.chunks;
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			// A traversal coordinate now represents a whole chunk or line.
			aPolicy.grain_size = 1;
		}
		if (layout.chunks > 1) {
			visit_coordinates(sequence_domain<TPolicy>{tasks},
				sequence_task<sequence_stage::totals, pass_t>{pass}, aPolicy);
			auto result = visit_coordinates(sequence_domain<TPolicy>{layout.lines},
				sequence_task<sequence_stage::carries, pass_t>{pass}, aPolicy);
			if constexpr (tKind == sequence_kind::reduction) {
				return result;
			}
		}
		return visit_coordinates(sequence_domain<TPolicy>{tasks},
			sequence_task<sequence_stage::output, pass_t>{pass}, aPolicy);
	}

	template<class TView>
	static void check_extent(typename TView::extents_t aActual, typename TView::extents_t aExpected)
	{
		for (int i = 0; i < TView::dimension; ++i) {
			if (math::detail::axis(aActual, i) != math::detail::axis(aExpected, i)) {
				throw std::invalid_argument("Sequence views must match the prepared shape");
			}
		}
	}
};

// Validate view shapes and select the work partition during preparation.
template<int tAxis, sequence_kind tKind, class TOp, class TValue>
struct sequence_algorithm {
	TOp op;
	TValue init;

	template<concepts::ReadableView TSrc, concepts::ViewDomain TDst, class TPolicy>
	requires concepts::WritableView<TDst, TValue> && concepts::SameViewBackend<TSrc, TDst> &&
		std::same_as<TPolicy, view_policy<TSrc>> && std::same_as<TPolicy, view_policy<TDst>> &&
		std::same_as<typename TDst::value_type, TValue> &&
		std::convertible_to<typename TSrc::value_type, TValue> &&
		std::is_trivially_copyable_v<TValue> && (alignof(TValue) <= alignof(std::max_align_t)) &&
		std::is_invocable_r_v<TValue, TOp&, TValue, TValue> && (tAxis < TSrc::dimension) &&
		(TDst::dimension == (tKind != sequence_kind::reduction ? TSrc::dimension :
			(tAxis < 0 || TSrc::dimension == 1 ? 1 : TSrc::dimension - 1)))
	auto prepare(TSrc aSource, TDst aDestination, TPolicy aPolicy) const
	{
		using namespace math::detail;
		const typename TSrc::extents_t extent = aSource.dimensions();
		const auto count = element_count<TSrc::dimension>(extent);
		typename TDst::extents_t expected{};
		std::size_t lines = 1;
		auto length = count;
		if constexpr (tAxis >= 0) {
			length = axis(extent, tAxis);
			auto line_extent = extent;
			axis(line_extent, tAxis) = 1;
			if (count != 0 || tKind == sequence_kind::reduction) {
				lines = element_count<TSrc::dimension>(line_extent);
			} else {
				lines = 0;
			}
		}
		if constexpr (tKind != sequence_kind::reduction) {
			for (int i = 0; i < TSrc::dimension; ++i) {
				axis(expected, i) = axis(extent, i);
			}
			if (count == 0) {
				lines = 0;
			}
		} else if constexpr (tAxis < 0 || TSrc::dimension == 1) {
			axis(expected, 0) = 1;
		} else {
			for (int i = 0; i < TDst::dimension; ++i) {
				axis(expected, i) = axis(extent, i + (i >= tAxis));
			}
		}
		using state_t = sequence_state<tAxis, tKind, TSrc, TDst, TOp, TValue, TPolicy>;
		state_t::template check_extent<TDst>(aDestination.dimensions(), expected);
		auto chunk_size = std::max<std::size_t>(1, length);
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			if (aPolicy.threads == 0 || aPolicy.grain_size == 0) {
				throw std::invalid_argument("CPU policy requires positive threads and grain size");
			}
			if (aPolicy.threads > 1 && lines < aPolicy.threads) {
				chunk_size = aPolicy.grain_size;
			}
		} else {
			chunk_size = 128;
		}
		const auto chunks = length == 0 || lines == 0 ? 1 : 1 + (length - 1) / chunk_size;
		return state_t{{extent, lines, length, chunk_size, chunks}, expected, op, init};
	}
};

} // namespace iane::detail
