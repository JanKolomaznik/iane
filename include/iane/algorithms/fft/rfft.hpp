#pragma once

#include <iane/algorithms/fft/fft.hpp>
#include <iane/algorithms/fft/layout.hpp>

#include <cstring>
#include <limits>

namespace iane {
namespace detail {

template<class TSrc, class TDst, class TPolicy, bool tInverse>
concept real_fft_views = concepts::MemoryView<TSrc> && concepts::MemoryView<TDst> &&
	concepts::TransformViews<TSrc, TDst> && TSrc::dimension == 2 &&
	std::same_as<typename TSrc::coord_t, math::vec2i> &&
	std::same_as<typename TSrc::extents_t, math::vec2i> &&
	std::same_as<typename TDst::extents_t, math::vec2i> &&
	std::same_as<typename TSrc::value_type, std::conditional_t<tInverse, fft_complex, float>> &&
	std::same_as<typename TDst::value_type, std::conditional_t<tInverse, float, fft_complex>> &&
	std::same_as<decltype(std::declval<TDst>().pointer()), typename TDst::value_type*> &&
	std::same_as<TPolicy, view_policy<TDst>>;

// Record the real shape and the byte count needed to preserve inverse input.
struct real_fft_layout {
	math::vec2i size;
	std::size_t input_copy_bytes;
};

template<class TPolicy, bool tInverse, class TSrc, class TDst>
real_fft_layout check_real_fft_views(TSrc aSource, TDst aDestination)
{
	const auto real = tInverse ? aDestination.dimensions() : aSource.dimensions();
	const auto packed = tInverse ? aSource.dimensions() : aDestination.dimensions();
	const auto expected = rfft_extent(real);
	if (packed[0] != expected[0] || packed[1] != expected[1]) {
		throw std::invalid_argument("Real FFT requires matching real and packed spectrum extents");
	}
	// cuFFT requires complex alignment even for its real buffers.
	constexpr auto real_alignment = std::same_as<TPolicy, cpu_policy> ? alignof(float) : alignof(fft_complex);
	const auto source_bytes = check_packed_fft_buffer(aSource,
		tInverse ? alignof(fft_complex) : real_alignment);
	const auto destination_bytes = check_packed_fft_buffer(aDestination,
		tInverse ? real_alignment : alignof(fft_complex));
	check_disjoint_fft_buffers(aSource, aDestination, source_bytes, destination_bytes);
	return {real, tInverse ? source_bytes : 0};
}

// Execute the reusable native plan with a private copy for destructive C2R backends.
template<class TNative, class TPolicy, bool tInverse>
struct real_fft_state {
	real_fft_layout layout;
	TNative native;

	std::size_t scratch_bytes() const
	{
		const auto work = native.scratch_bytes();
		if (work > std::numeric_limits<std::size_t>::max() - layout.input_copy_bytes) {
			throw std::overflow_error("Real FFT scratch size overflows");
		}
		return layout.input_copy_bytes + work;
	}

	template<class TSrc, class TDst>
	requires real_fft_views<TSrc, TDst, TPolicy, tInverse>
	auto run(TSrc aSource, TDst aDestination, scratch_space<TPolicy>& aScratch, TPolicy aPolicy)
	{
		const auto actual = check_real_fft_views<TPolicy, tInverse>(aSource, aDestination);
		if (actual.size[0] != layout.size[0] || actual.size[1] != layout.size[1]) {
			throw std::invalid_argument("Real FFT views must match the prepared shape");
		}
		auto* input = const_cast<typename TSrc::value_type*>(aSource.pointer());
		if constexpr (tInverse) {
			// Allocate this first. Its size is a multiple of the cuFFT work area's alignment.
			input = static_cast<fft_complex*>(aScratch.bytes(layout.input_copy_bytes, alignof(fft_complex)));
			if constexpr (std::same_as<TPolicy, cpu_policy>) {
				std::memcpy(input, aSource.pointer(), layout.input_copy_bytes);
#if defined(__CUDACC__)
			} else {
				check_cuda(cudaMemcpyAsync(input, aSource.pointer(), layout.input_copy_bytes,
					cudaMemcpyDeviceToDevice, aPolicy.stream), "Copy inverse FFT input");
#endif
			}
		}
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			if constexpr (tInverse) {
				fftwf_execute_dft_c2r(native.plan.get(), fftw_state::native(input), aDestination.pointer());
			} else {
				fftwf_execute_dft_r2c(native.plan.get(), input, fftw_state::native(aDestination.pointer()));
			}
			return cpu_execution{};
#if defined(__CUDACC__)
		} else {
			native.bind_workspace(aScratch);
			if constexpr (tInverse) {
				check_cufft(cufftExecC2R(native.plan->handle, cufft_state::native(input),
					aDestination.pointer()), "Execute inverse CUDA real FFT");
			} else {
				check_cufft(cufftExecR2C(native.plan->handle, input,
					cufft_state::native(aDestination.pointer())), "Execute CUDA real FFT");
			}
			return cuda_execution{aPolicy.stream};
#endif
		}
	}
};

// Plan real-to-complex or complex-to-real execution from the real extent.
template<bool tInverse>
struct real_fft_algorithm {
	template<class TSrc, class TDst, class TPolicy>
	requires real_fft_views<TSrc, TDst, TPolicy, tInverse>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy aPolicy) const
	{
		const auto layout = check_real_fft_views<TPolicy, tInverse>(aSource, aDestination);
		const auto size = layout.size;
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			fftwf_plan plan;
			{
				const std::lock_guard lock(fftw_planner_mutex);
				constexpr unsigned flags = FFTW_ESTIMATE | FFTW_UNALIGNED;
				if constexpr (tInverse) {
					plan = fftwf_plan_dft_c2r_2d(size[1], size[0], fftw_state::native(aSource.pointer()),
						aDestination.pointer(), flags | FFTW_DESTROY_INPUT);
				} else {
					plan = fftwf_plan_dft_r2c_2d(size[1], size[0], const_cast<float*>(aSource.pointer()),
						fftw_state::native(aDestination.pointer()), flags | FFTW_PRESERVE_INPUT);
				}
			}
			if (!plan) {
				throw std::runtime_error("Create FFTW real transform plan failed");
			}
			return real_fft_state<fftw_state, TPolicy, tInverse>{layout, {fftw_state::plan_t(plan)}};
#if defined(__CUDACC__)
		} else {
			return real_fft_state<cufft_state, TPolicy, tInverse>{layout,
				make_cufft_state(size, tInverse ? CUFFT_C2R : CUFFT_R2C,
					tInverse ? CUFFT_INVERSE : CUFFT_FORWARD, aPolicy)};
#endif
		}
	}
};

} // namespace detail

/**
 * @brief Prepare an out-of-place 2D real FFT from float pixels to fft_complex bins.
 *
 * Both views must be packed and disjoint. The spectrum extent is rfft_extent
 * of the real input. CUDA pointers require complex alignment. The forward
 * transform uses the negative exponent and preserves its input.
 */
inline constexpr detail::real_fft_algorithm<false> rfft{};

/**
 * @brief Prepare an unnormalized 2D inverse real FFT from a packed Hermitian spectrum.
 *
 * The real destination extent determines the original shape. Packed layout,
 * alignment and disjointness requirements are the same as rfft. The context
 * reserves a packed-spectrum copy so the supplied input remains intact.
 */
inline constexpr detail::real_fft_algorithm<true> irfft{};

} // namespace iane
