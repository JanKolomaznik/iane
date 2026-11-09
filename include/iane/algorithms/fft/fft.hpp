#pragma once

#include <iane/algorithms/fft/complex.hpp>
#include <iane/algorithms/transform.hpp>
#include <iane/execution/prepared.hpp>

#include <fftw3.h>
#if defined(__CUDACC__)
#include <cufft.h>
#endif

#include <cstdint>
#include <mutex>
#include <string>

namespace iane {

namespace detail {

template<class TSrc, class TDst, class TPolicy>
concept fft_views = concepts::MemoryView<TSrc> && concepts::MemoryView<TDst> &&
	concepts::TransformViews<TSrc, TDst> && TSrc::dimension == 2 &&
	std::same_as<typename TSrc::coord_t, math::vec2i> &&
	std::same_as<typename TSrc::value_type, fft_complex> &&
	std::same_as<decltype(std::declval<TDst>().pointer()), fft_complex*> &&
	std::same_as<TPolicy, view_policy<TDst>>;

template<class TView>
std::size_t check_packed_fft_buffer(TView aView, std::size_t aAlignment)
{
	using element_t = typename TView::value_type;
	const auto [count, strides] = resource::detail::packed_layout<element_t, 2>(aView.dimensions());
	const auto actual = aView.strides();
	if (actual[0] != strides[0] || actual[1] != strides[1]) {
		throw std::invalid_argument("FFT requires packed views");
	}
	const auto address = reinterpret_cast<std::uintptr_t>(aView.pointer());
	if (address == 0 || address % aAlignment != 0) {
		throw std::invalid_argument("FFT requires aligned input and output buffers");
	}
	return count * sizeof(element_t);
}

template<class TSrc, class TDst>
void check_disjoint_fft_buffers(TSrc aSource, TDst aDestination,
	std::size_t aSourceBytes, std::size_t aDestinationBytes)
{
	const auto src = reinterpret_cast<std::uintptr_t>(aSource.pointer());
	const auto dst = reinterpret_cast<std::uintptr_t>(aDestination.pointer());
	if (src <= dst ? dst - src < aSourceBytes : src - dst < aDestinationBytes) {
		throw std::invalid_argument("FFT requires disjoint input and output buffers");
	}
}

template<class TSrc, class TDst>
math::vec2i check_fft_views(TSrc aSource, TDst aDestination)
{
	const auto size = matching_dimensions(aSource, aDestination);
	if (size[0] <= 0 || size[1] <= 0) {
		throw std::invalid_argument("FFT requires positive dimensions");
	}
	const auto source_bytes = check_packed_fft_buffer(aSource, alignof(fft_complex));
	const auto destination_bytes = check_packed_fft_buffer(aDestination, alignof(fft_complex));
	check_disjoint_fft_buffers(aSource, aDestination, source_bytes, destination_bytes);
	return size;
}

inline std::mutex fftw_planner_mutex;

// Serialize FFTW plan destruction with creation in IANE.
struct fftw_plan_deleter {
	void operator()(fftwf_plan aPlan) const
	{
		const std::lock_guard lock(fftw_planner_mutex);
		fftwf_destroy_plan(aPlan);
	}
};

// Hold an FFTW plan which can execute on new packed buffers.
struct fftw_state {
	using plan_t = std::unique_ptr<std::remove_pointer_t<fftwf_plan>, fftw_plan_deleter>;
	plan_t plan;

	std::size_t scratch_bytes() const { return 0; }

	cpu_execution run(const fft_complex* aSource, fft_complex* aDestination,
		scratch_space<cpu_policy>&)
	{
		fftwf_execute_dft(plan.get(), native(aSource), native(aDestination));
		return {};
	}

	static fftwf_complex* native(const fft_complex* aPointer)
	{
		static_assert(sizeof(fft_complex) == sizeof(fftwf_complex));
		static_assert(offsetof(fft_complex, imag) == sizeof(float));
		return reinterpret_cast<fftwf_complex*>(const_cast<fft_complex*>(aPointer));
	}
};

#if defined(__CUDACC__)
inline void check_cufft(cufftResult aStatus, const char* aOperation)
{
	if (aStatus != CUFFT_SUCCESS) {
		throw std::runtime_error(std::string(aOperation) + " (cuFFT status " +
			std::to_string(static_cast<int>(aStatus)) + ")");
	}
}

// Own a cuFFT plan bound to its context's stream and protect in-flight execution.
struct cufft_plan {
	cufftHandle handle{};
	cudaStream_t stream{};
	bool submitted{};

	~cufft_plan()
	{
		if (submitted) {
			cudaStreamSynchronize(stream);
		}
		if (handle) {
			cufftDestroy(handle);
		}
	}
};

// Bind the shared work area before the first execution and whenever it grows.
struct cufft_state {
	std::unique_ptr<cufft_plan> plan;
	std::size_t bytes;
	int direction;
	void* work_area{};

	std::size_t scratch_bytes() const { return bytes; }

	void bind_workspace(scratch_space<cuda_policy>& aScratch)
	{
		auto* area = aScratch.bytes(bytes, alignof(cufftComplex));
		if (area != work_area) {
			check_cufft(cufftSetWorkArea(plan->handle, area), "Set FFT workspace");
			work_area = area;
		}
		plan->submitted = true;
	}

	static cufftComplex* native(const fft_complex* aPointer)
	{
		static_assert(sizeof(fft_complex) == sizeof(cufftComplex));
		static_assert(alignof(fft_complex) >= alignof(cufftComplex));
		return reinterpret_cast<cufftComplex*>(const_cast<fft_complex*>(aPointer));
	}

	cuda_execution run(const fft_complex* aSource, fft_complex* aDestination,
		scratch_space<cuda_policy>& aScratch)
	{
		bind_workspace(aScratch);
		check_cufft(cufftExecC2C(plan->handle,
			native(aSource), native(aDestination), direction), "Execute CUDA FFT");
		return {plan->stream};
	}
};
inline cufft_state make_cufft_state(math::vec2i aSize, cufftType aType, int aDirection, cuda_policy aPolicy)
{
	auto plan = std::make_unique<cufft_plan>();
	plan->stream = aPolicy.stream;
	check_cufft(cufftCreate(&plan->handle), "Create CUDA FFT plan");
	check_cufft(cufftSetAutoAllocation(plan->handle, 0), "Disable CUDA FFT auto allocation");
	std::size_t bytes{};
	check_cufft(cufftMakePlan2d(plan->handle, aSize[1], aSize[0], aType, &bytes), "Plan CUDA FFT");
	check_cufft(cufftSetStream(plan->handle, plan->stream), "Set CUDA FFT stream");
	return {std::move(plan), bytes, aDirection};
}
#endif

// Validate new views before executing the backend's reusable FFT plan.
template<class TNative, class TBackendPolicy>
struct fft_state {
	math::vec2i size;
	TNative native;

	std::size_t scratch_bytes() const { return native.scratch_bytes(); }

	template<class TSrc, class TDst>
	requires fft_views<TSrc, TDst, TBackendPolicy>
	auto run(TSrc aSource, TDst aDestination, scratch_space<TBackendPolicy>& aScratch, TBackendPolicy)
	{
		const auto extent = check_fft_views(aSource, aDestination);
		if (extent[0] != size[0] || extent[1] != size[1]) {
			throw std::invalid_argument("FFT views must match the prepared shape");
		}
		return native.run(aSource.pointer(), aDestination.pointer(), aScratch);
	}
};

// Construct a backend plan from the packed prototype layout.
struct fft_algorithm {
	bool inverse;

	template<class TSrc, class TDst, class TPolicy>
	requires fft_views<TSrc, TDst, TPolicy>
	auto prepare(TSrc aSource, TDst aDestination, TPolicy aPolicy) const
	{
		const auto size = check_fft_views(aSource, aDestination);
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			fftwf_plan plan;
			{
				const std::lock_guard lock(fftw_planner_mutex);
				plan = fftwf_plan_dft_2d(size[1], size[0], fftw_state::native(aSource.pointer()),
					fftw_state::native(aDestination.pointer()), inverse ? FFTW_BACKWARD : FFTW_FORWARD,
					FFTW_ESTIMATE | FFTW_UNALIGNED | FFTW_PRESERVE_INPUT);
			}
			if (!plan) {
				throw std::runtime_error("Create FFTW plan failed");
			}
			return fft_state<fftw_state, cpu_policy>{size, {fftw_state::plan_t(plan)}};
#if defined(__CUDACC__)
		} else {
			return fft_state<cufft_state, cuda_policy>{size,
				make_cufft_state(size, CUFFT_C2C, inverse ? CUFFT_INVERSE : CUFFT_FORWARD, aPolicy)};
#endif
		}
	}
};

} // namespace detail

/// Prepare an out-of-place packed 2D complex FFT with a negative exponent.
inline constexpr detail::fft_algorithm fft_forward{false};

/// Prepare the unnormalized inverse of fft_forward.
inline constexpr detail::fft_algorithm fft_inverse{true};

} // namespace iane
