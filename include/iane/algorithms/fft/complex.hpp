#pragma once

namespace iane {

/// Single-precision complex pixel in interleaved real/imaginary order.
struct alignas(2 * sizeof(float)) fft_complex {
	float real;
	float imag;
};

} // namespace iane
