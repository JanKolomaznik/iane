#pragma once

#include <iane/algorithms/transform.hpp>
#include <iane/math/offset.hpp>

#include <concepts>
#include <stdexcept>

namespace iane {

namespace detail {

// Forward the source pixel unchanged for same-backend copies through transform.
struct copy_value {
	template<class TElement>
	IANE_DECL_HYBRID TElement&& operator()(TElement&& aValue) const { return static_cast<TElement&&>(aValue); }
};

#if defined(__CUDACC__)
template<class TView>
concept cuda_transfer_view = requires(TView aView) { aView.pointer(); aView.strides(); };
#endif

} // namespace detail

/**
 * @brief Materialize a readable host view into a writable host view.
 * @return An already-completed result.
 * @throws std::invalid_argument If the dimensions differ or are negative.
 * @pre Source storage must remain readable throughout the copy. Shifted
 * overlapping views are unsupported.
 */
template<concepts::HostView TSrc, concepts::HostView TDst>
requires (TSrc::dimension > 0 && TSrc::dimension == TDst::dimension)
cpu_execution copy(TSrc aSource, TDst aDestination, cpu_execution = {})
{
	return transform(aSource, aDestination, detail::copy_value{});
}

#if defined(__CUDACC__)
/**
 * @brief Enqueue materialization of a readable CUDA view on the given stream.
 * @return The execution handle for waiting on the submitted work.
 * @pre Source and destination storage must remain valid until completion.
 * Shifted overlapping views are unsupported.
 */
template<concepts::DeviceView TSrc, concepts::DeviceView TDst>
requires (TSrc::dimension > 0 && TSrc::dimension == TDst::dimension)
cuda_execution copy(TSrc aSource, TDst aDestination, cuda_execution aExecution = {})
{
	return transform(aSource, aDestination, detail::copy_value{}, aExecution);
}

/**
 * @brief Enqueue a host/device transfer with matching element types.
 * @return The execution handle for waiting on the transfer.
 * @pre Rows must contain contiguous elements. Both views must remain valid
 * until completion. CUDA may synchronize pageable host transfers internally.
 */
template<detail::cuda_transfer_view TSrc, detail::cuda_transfer_view TDst>
requires (TSrc::dimension > 0 && TSrc::dimension == TDst::dimension &&
	((concepts::HostView<TSrc> && concepts::DeviceView<TDst>) ||
	 (concepts::DeviceView<TSrc> && concepts::HostView<TDst>)) &&
	std::same_as<typename TSrc::value_type, typename TDst::value_type> &&
	std::is_trivially_copyable_v<typename TSrc::value_type>)
cuda_execution copy(TSrc aSource, TDst aDestination, cuda_execution aExecution = {})
{
	const auto size = detail::matching_dimensions(aSource, aDestination);
	constexpr int dimension = TSrc::dimension;
	const auto count = math::detail::element_count<dimension>(size);
	if (count == 0) {
		return aExecution;
	}
	constexpr auto element_size = sizeof(typename TSrc::value_type);
	const auto width = static_cast<std::size_t>(math::detail::axis(size, 0));
	if (count > std::numeric_limits<std::size_t>::max() / element_size) {
		throw std::overflow_error("CUDA transfer byte size overflows");
	}
	const auto row_bytes = width * element_size;
	const auto src_stride = aSource.strides();
	const auto dst_stride = aDestination.strides();
	if (math::detail::axis(src_stride, 0) != element_size ||
	    math::detail::axis(dst_stride, 0) != element_size) {
		throw std::invalid_argument("CUDA transfer requires contiguous elements within each row");
	}
	constexpr auto direction = concepts::HostView<TSrc> ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
	std::size_t packed_stride = element_size;
	bool packed = true;
	for (int i = 0; i < dimension; ++i) {
		packed &= static_cast<std::size_t>(math::detail::axis(src_stride, i)) == packed_stride &&
			static_cast<std::size_t>(math::detail::axis(dst_stride, i)) == packed_stride;
		packed_stride *= static_cast<std::size_t>(math::detail::axis(size, i));
	}
	if (packed) {
		detail::check_cuda(cudaMemcpyAsync(aDestination.pointer(), aSource.pointer(),
			count * element_size, direction, aExecution.stream), "CUDA image transfer");
		return aExecution;
	}
	if constexpr (dimension > 1) {
		if (src_stride[1] < 0 || dst_stride[1] < 0 ||
		    static_cast<std::size_t>(src_stride[1]) < row_bytes ||
		    static_cast<std::size_t>(dst_stride[1]) < row_bytes) {
			throw std::invalid_argument("CUDA transfer requires a valid row pitch");
		}
		const auto plane_count = width * size[1];
		auto* destination = reinterpret_cast<char*>(aDestination.pointer());
		const auto* source = reinterpret_cast<const char*>(aSource.pointer());
		for (std::size_t index = 0; index < count; index += plane_count) {
			const auto position = math::detail::coordinate_at<dimension, typename TSrc::coord_t>(index, size);
			detail::check_cuda(cudaMemcpy2DAsync(
				destination + math::offset_in_strided_memory(dst_stride, position), dst_stride[1],
				source + math::offset_in_strided_memory(src_stride, position), src_stride[1],
				row_bytes, size[1], direction, aExecution.stream), "CUDA image transfer");
		}
	}
	return aExecution;
}
#endif

} // namespace iane
