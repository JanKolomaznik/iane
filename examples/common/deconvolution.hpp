#pragma once

#include "kernel.hpp"

#include <iane/execution/run_on.hpp>

namespace deconvolution {

/// Read a nonnegative PSF and normalize its coefficient sum to one.
inline auto read_psf(const std::string& aPath)
{
	auto psf = io::read_kernel(aPath);
	const auto size = psf.dimensions();
	const auto count = static_cast<std::size_t>(size[0]) * size[1];
	double sum = 0.0;
	for (std::size_t i = 0; i < count; ++i) {
		const float coefficient = psf.pointer()[i];
		if (coefficient < 0.0f) {
			throw std::invalid_argument("PSF coefficients must be nonnegative");
		}
		sum += coefficient;
	}
	if (!(sum > 0.0)) {
		throw std::invalid_argument("PSF must have a positive coefficient sum");
	}
	for (std::size_t i = 0; i < count; ++i) {
		psf.pointer()[i] = static_cast<float>(psf.pointer()[i] / sum);
	}
	return psf;
}

template<class TElement, class TView>
using backend_image = std::conditional_t<iane::concepts::HostView<TView>,
	iane::host_image<TElement, 2>,
#if defined(__CUDACC__)
	iane::device_image<TElement, 2>
#else
	iane::host_image<TElement, 2>
#endif
>;

// Convert a restored intensity in byte units to a grayscale output pixel.
struct to_gray8 {
	IANE_DECL_HYBRID std::uint8_t operator()(float aValue) const
	{
		if (!(aValue > 0.0f)) {
			return 0;
		}
		if (aValue >= 255.0f) {
			return 255;
		}
		const auto integer = static_cast<unsigned int>(aValue);
		return static_cast<std::uint8_t>(integer + (aValue - integer >= 0.5f));
	}
};

} // namespace deconvolution
