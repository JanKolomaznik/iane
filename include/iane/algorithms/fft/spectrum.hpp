#pragma once

#include <iane/algorithms/fft/complex.hpp>
#include <iane/algorithms/fft/layout.hpp>
#include <iane/view/algebra.hpp>

namespace iane {
namespace detail {

template<class TView>
concept spectrum_view = algebra_view<TView> && TView::dimension == 2 &&
	std::same_as<typename TView::coord_t, math::vec2i> &&
	std::same_as<typename TView::extents_t, math::vec2i> &&
	std::same_as<typename TView::value_type, fft_complex>;

// Translate centered display coordinates to the unshifted spectrum.
template<class TView>
struct shifted_spectrum_pixel {
	TView spectrum;
	math::vec2i size;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID fft_complex operator()(math::vec2i aPosition) const
	{
		for (int axis = 0; axis < 2; ++axis) {
			const int center = size[axis] / 2;
			aPosition[axis] = aPosition[axis] < center ?
				aPosition[axis] + (size[axis] - center) : aPosition[axis] - center;
		}
		return spectrum[aPosition];
	}
};

// Reconstruct omitted frequencies from their conjugate partners.
template<class TView>
struct hermitian_spectrum_pixel {
	TView spectrum;
	math::vec2i size;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID fft_complex operator()(math::vec2i aPosition) const
	{
		if (aPosition[0] <= size[0] / 2) {
			return spectrum[aPosition];
		}
		aPosition[0] = size[0] - aPosition[0];
		if (aPosition[1] != 0) {
			aPosition[1] = size[1] - aPosition[1];
		}
		auto pixel = static_cast<fft_complex>(spectrum[aPosition]);
		pixel.imag = -pixel.imag;
		return pixel;
	}
};

// Compute a complex magnitude without squaring the components directly.
struct magnitude_pixel {
	IANE_DECL_HYBRID float operator()(fft_complex aPixel) const
	{
		return std::hypot(aPixel.real, aPixel.imag);
	}
};

} // namespace detail

/**
 * @brief Borrow a full spectrum with DC centered at floor(extent / 2).
 * @param aSpectrum Readable unshifted complex spectrum on either backend.
 * @return Lazy shifted view with unchanged extents. Keep the spectrum alive through use.
 */
template<detail::spectrum_view TView>
auto fftshift_view(TView aSpectrum)
{
	return generate_view(aSpectrum, detail::shifted_spectrum_pixel<TView>{aSpectrum, aSpectrum.dimensions()});
}

/**
 * @brief Borrow the full spectrum of a real image from its packed axis-zero half.
 * @param aSpectrum Unshifted bins with extent {width / 2 + 1, height}.
 * @param aFullExtent Positive original width and height.
 * @return Lazy full view on the source backend. Keep the source alive through use.
 * @throws std::invalid_argument For inconsistent extents.
 */
template<detail::spectrum_view TView>
auto hermitian_spectrum_view(TView aSpectrum, math::vec2i aFullExtent)
{
	const auto packed = aSpectrum.dimensions();
	const auto expected = rfft_extent(aFullExtent);
	if (packed[0] != expected[0] || packed[1] != expected[1]) {
		throw std::invalid_argument("Hermitian spectrum requires matching packed and full extents");
	}
	using pixel_t = detail::hermitian_spectrum_pixel<TView>;
	return detail::procedural_view<TView, pixel_t>{{}, aFullExtent, {aSpectrum, aFullExtent}};
}

/// Lazily return the magnitude of each complex bin. See map_view for lifetimes.
template<detail::spectrum_view TView>
auto magnitude_view(TView aSpectrum)
{
	return map_view(aSpectrum, detail::magnitude_pixel{});
}

/// Lazily return log(1 + magnitude) for display. See magnitude_view for lifetimes.
template<detail::spectrum_view TView>
auto log_magnitude_view(TView aSpectrum)
{
	return iane::log1p(magnitude_view(aSpectrum));
}

} // namespace iane
