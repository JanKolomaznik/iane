#pragma once

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

/// Single-precision complex pixel in interleaved real/imaginary order.
struct alignas(2 * sizeof(float)) fft_complex {
	float real;
	float imag;
};

namespace detail {

template<class TSrc, class TDst, class TPolicy>
concept fft_views = concepts::MemoryView<TSrc> && concepts::MemoryView<TDst> &&
	concepts::TransformViews<TSrc, TDst> && TSrc::dimension == 2 &&
	std::same_as<typename TSrc::coord_t, math::vec2i> &&
	std::same_as<typename TSrc::value_type, fft_complex> &&
	std::same_as<decltype(std::declval<TDst>().pointer()), fft_complex*> &&
	std::same_as<TPolicy, view_policy<TDst>>;

template<class TSrc, class TDst>
math::vec2i check_fft_views(TSrc aSource, TDst aDestination)
{
	const auto size = matching_dimensions(aSource, aDestination);
	if (size[0] <= 0 || size[1] <= 0) {
		throw std::invalid_argument("FFT requires positive dimensions");
	}
	const auto [count, strides] = resource::detail::packed_layout<fft_complex, 2>(size);
	const auto source_strides = aSource.strides();
	const auto destination_strides = aDestination.strides();
	for (int axis = 0; axis < 2; ++axis) {
		if (source_strides[axis] != strides[axis] || destination_strides[axis] != strides[axis]) {
			throw std::invalid_argument("FFT requires packed views");
		}
	}
	const auto src = reinterpret_cast<std::uintptr_t>(aSource.pointer());
	const auto dst = reinterpret_cast<std::uintptr_t>(aDestination.pointer());
	const auto distance = src < dst ? dst - src : src - dst;
	if (src == 0 || dst == 0 || src % alignof(fft_complex) != 0 ||
	    dst % alignof(fft_complex) != 0 || distance < count * sizeof(fft_complex)) {
		throw std::invalid_argument("FFT requires aligned, disjoint input and output buffers");
	}
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

	cuda_execution run(const fft_complex* aSource, fft_complex* aDestination,
		scratch_space<cuda_policy>& aScratch)
	{
		static_assert(sizeof(fft_complex) == sizeof(cufftComplex));
		static_assert(alignof(fft_complex) >= alignof(cufftComplex));
		auto* area = aScratch.bytes(bytes, alignof(cufftComplex));
		if (area != work_area) {
			check_cufft(cufftSetWorkArea(plan->handle, area), "Set FFT workspace");
			work_area = area;
		}
		plan->submitted = true;
		check_cufft(cufftExecC2C(plan->handle,
			reinterpret_cast<cufftComplex*>(const_cast<fft_complex*>(aSource)),
			reinterpret_cast<cufftComplex*>(aDestination), direction), "Execute CUDA FFT");
		return {plan->stream};
	}
};
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
			auto plan = std::make_unique<cufft_plan>();
			plan->stream = aPolicy.stream;
			check_cufft(cufftCreate(&plan->handle), "Create CUDA FFT plan");
			check_cufft(cufftSetAutoAllocation(plan->handle, 0), "Disable CUDA FFT auto allocation");
			std::size_t bytes{};
			check_cufft(cufftMakePlan2d(plan->handle, size[1], size[0], CUFFT_C2C, &bytes),
				"Plan CUDA FFT");
			check_cufft(cufftSetStream(plan->handle, plan->stream), "Set CUDA FFT stream");
			return fft_state<cufft_state, cuda_policy>{size,
				{std::move(plan), bytes, inverse ? CUFFT_INVERSE : CUFFT_FORWARD}};
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
