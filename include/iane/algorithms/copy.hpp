#pragma once

#include <iane/algorithms/transform.hpp>

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
 * @brief Materialize a readable 2D host view into a writable host view.
 * @return An already-completed result.
 * @throws std::invalid_argument If the dimensions differ or are negative.
 * @pre Source storage must remain readable throughout the copy. Shifted
 * overlapping views are unsupported.
 */
template<concepts::HostView TSrc, concepts::HostView TDst>
requires (TSrc::dimension == 2 && TDst::dimension == 2)
cpu_execution copy(TSrc aSource, TDst aDestination, cpu_execution = {})
{
	return transform(aSource, aDestination, detail::copy_value{});
}

#if defined(__CUDACC__)
/**
 * @brief Enqueue materialization of a readable 2D CUDA view on the given stream.
 * @return The execution handle for waiting on the submitted work.
 * @pre Source and destination storage must remain valid until completion.
 * Shifted overlapping views are unsupported.
 */
template<concepts::DeviceView TSrc, concepts::DeviceView TDst>
requires (TSrc::dimension == 2 && TDst::dimension == 2)
cuda_execution copy(TSrc aSource, TDst aDestination, cuda_execution aExecution = {})
{
	return transform(aSource, aDestination, detail::copy_value{}, aExecution);
}

/**
 * @brief Enqueue a 2D host/device transfer with matching element types.
 * @return The execution handle for waiting on the transfer.
 * @pre Rows must contain contiguous elements. Both views must remain valid
 * until completion. CUDA may synchronize pageable host transfers internally.
 */
template<detail::cuda_transfer_view TSrc, detail::cuda_transfer_view TDst>
requires (TSrc::dimension == 2 && TDst::dimension == 2 &&
	((concepts::HostView<TSrc> && concepts::DeviceView<TDst>) ||
	 (concepts::DeviceView<TSrc> && concepts::HostView<TDst>)) &&
	std::same_as<typename TSrc::value_type, typename TDst::value_type>)
cuda_execution copy(TSrc aSource, TDst aDestination, cuda_execution aExecution = {})
{
	const auto size = detail::matching_dimensions(aSource, aDestination);
	if (size[0] == 0 || size[1] == 0) {
		return aExecution;
	}
	constexpr auto element_size = sizeof(typename TSrc::value_type);
	const auto row_bytes = static_cast<std::size_t>(size[0]) * element_size;
	const auto src_stride = aSource.strides();
	const auto dst_stride = aDestination.strides();
	if (src_stride[0] != element_size || dst_stride[0] != element_size ||
	    src_stride[1] < 0 || dst_stride[1] < 0 ||
	    static_cast<std::size_t>(src_stride[1]) < row_bytes ||
	    static_cast<std::size_t>(dst_stride[1]) < row_bytes) {
		throw std::invalid_argument("CUDA transfer requires contiguous elements within each row");
	}
	constexpr auto direction = concepts::HostView<TSrc> ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
	detail::check_cuda(cudaMemcpy2DAsync(aDestination.pointer(), dst_stride[1],
		aSource.pointer(), src_stride[1], row_bytes, size[1], direction, aExecution.stream),
		"CUDA image transfer");
	return aExecution;
}
#endif

} // namespace iane
