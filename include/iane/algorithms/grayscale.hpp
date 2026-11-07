#pragma once

#include <iane/concepts/view.hpp>
#include <iane/config.hpp>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace iane {

/// Weights applied to the stored red, green and blue channel values.
struct rgb_weights {
	double red;
	double green;
	double blue;
};

namespace detail {

template<class TView>
struct grayscale_view {
	using value_type = std::uint8_t;
	using coord_t = typename TView::coord_t;
	static constexpr int dimension = TView::dimension;

	TView source;
	rgb_weights weights;

	auto dimensions() const { return source.dimensions(); }

	template<class TPixel>
	IANE_DECL_HYBRID value_type convert(TPixel aPixel) const
	{
		const double value = ::fma(weights.red, aPixel.r,
			::fma(weights.green, aPixel.g, weights.blue * aPixel.b));
		if (value <= 0.0) {
			return 0;
		}
		if (value >= 255.0) {
			return 255;
		}
		const auto integer = static_cast<unsigned int>(value);
		return static_cast<value_type>(integer + (value - integer >= 0.5));
	}
};

template<class TView>
struct host_grayscale_view : grayscale_view<TView> {
	using is_host_view_tag = void;

	std::uint8_t operator[](const typename TView::coord_t& aPosition) const
	{
		return this->convert(this->source[aPosition]);
	}
};

template<class TView>
struct device_grayscale_view : grayscale_view<TView> {
	using is_device_view_tag = void;

	IANE_DECL_HYBRID auto dimensions() const { return this->source.dimensions(); }

	IANE_CUDA_DEVICE std::uint8_t operator[](const typename TView::coord_t& aPosition) const
	{
		return this->convert(this->source[aPosition]);
	}
};

} // namespace detail

/**
 * @brief Describe an RGB to gray8 conversion without allocating storage.
 * @param aSource Readable view of pixels with byte-valued r, g and b channels.
 * @param aWeights Channel weights.
 * @return Readable view borrowing source storage. Values round to nearest with
 * ties toward positive infinity, then clamp to the byte range.
 * @throws std::invalid_argument If weights are nonfinite or can overflow the sum.
 */
template<concepts::AnyView TView>
auto grayscale(TView aSource, rgb_weights aWeights)
{
	constexpr auto limit = std::numeric_limits<double>::max() / 1024;
	if (!(std::abs(aWeights.red) <= limit && std::abs(aWeights.green) <= limit &&
	      std::abs(aWeights.blue) <= limit)) {
		throw std::invalid_argument("Grayscale weights are outside the supported range");
	}
	if constexpr (concepts::HostView<TView>) {
		return detail::host_grayscale_view<TView>{{aSource, aWeights}};
	} else {
		return detail::device_grayscale_view<TView>{{aSource, aWeights}};
	}
}

} // namespace iane
