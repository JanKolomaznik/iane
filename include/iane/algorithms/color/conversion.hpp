#pragma once

#include <iane/concepts/element.hpp>
#include <iane/math/coord_traits.hpp>

#include <cmath>

namespace iane {
namespace detail {

// Normalize byte channels independently of their color-space interpretation.
struct normalize_rgb8 {
	template<concepts::Rgb8Element TPixel>
	IANE_DECL_HYBRID math::vec3f operator()(TPixel aPixel) const
	{
		return math::vec3f{aPixel.r / 255.0f, aPixel.g / 255.0f, aPixel.b / 255.0f};
	}
};

// Decode each sRGB channel into linear light.
struct decode_srgb {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aRgb) const
	{
		for (int i = 0; i < 3; ++i) {
			const auto value = std::abs(aRgb[i]);
			const auto linear = value <= 0.04045f ? value / 12.92f :
				std::pow((value + 0.055f) / 1.055f, 2.4f);
			aRgb[i] = std::copysign(linear, aRgb[i]);
		}
		return aRgb;
	}
};

// Encode each linear-light channel with the sRGB transfer curve.
struct encode_srgb {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aRgb) const
	{
		for (int i = 0; i < 3; ++i) {
			const auto value = std::abs(aRgb[i]);
			const auto encoded = value <= 0.0031308f ? 12.92f * value :
				1.055f * std::pow(value, 1.0f / 2.4f) - 0.055f;
			aRgb[i] = std::copysign(encoded, aRgb[i]);
		}
		return aRgb;
	}
};

// Extract HSV using the dominant RGB channel to select the hue sector.
struct rgb_to_hsv_pixel {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aRgb) const
	{
		const auto [red, green, blue] = aRgb.mElements;
		const auto maximum = std::fmax(red, std::fmax(green, blue));
		const auto range = maximum - std::fmin(red, std::fmin(green, blue));
		if (range == 0.0f) {
			return math::vec3f{0.0f, 0.0f, maximum};
		}
		float hue;
		if (maximum == red) {
			hue = (green - blue) / range;
		} else if (maximum == green) {
			hue = 2.0f + (blue - red) / range;
		} else {
			hue = 4.0f + (red - green) / range;
		}
		hue /= 6.0f;
		hue -= std::floor(hue);
		return math::vec3f{hue < 1.0f ? hue : 0.0f, range / maximum, maximum};
	}
};

// Interpolate RGB channels within the wrapped HSV hue sector.
struct hsv_to_rgb_pixel {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aHsv) const
	{
		const auto [hue, saturation, value] = aHsv.mElements;
		const float sector = 6.0f * (hue - std::floor(hue));
		const float chroma = value * saturation;
		math::vec3f rgb{};
		for (int i = 0; i < 3; ++i) {
			const float k = std::fmod(sector + (i == 0 ? 5.0f : i == 1 ? 3.0f : 1.0f), 6.0f);
			const float ramp = std::fmax(0.0f, std::fmin(1.0f, std::fmin(k, 4.0f - k)));
			rgb[i] = value - chroma * ramp;
		}
		return rgb;
	}
};

// Compute relative luminance with the Y row of the sRGB-to-XYZ matrix.
struct linear_luminance {
	IANE_DECL_HYBRID float operator()(math::vec3f aRgb) const
	{
		return (87098.0f / 409605) * aRgb[0] + (175762.0f / 245763) * aRgb[1] +
			(12673.0f / 175545) * aRgb[2];
	}
};

// Apply the D65 linear sRGB-to-XYZ matrix.
struct linear_rgb_to_xyz_pixel {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aRgb) const
	{
		const auto [red, green, blue] = aRgb.mElements;
		return math::vec3f{
			(506752.0f / 1228815) * red + (87881.0f / 245763) * green + (12673.0f / 70218) * blue,
			linear_luminance{}(aRgb),
			(7918.0f / 409605) * red + (87881.0f / 737289) * green + (1001167.0f / 1053270) * blue};
	}
};

// Apply the inverse D65 XYZ-to-linear sRGB matrix.
struct xyz_to_linear_rgb_pixel {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aXyz) const
	{
		const auto [x, y, z] = aXyz.mElements;
		return math::vec3f{
			(12831.0f / 3959) * x - (329.0f / 214) * y - (1974.0f / 3959) * z,
			-(851781.0f / 878810) * x + (1648619.0f / 878810) * y + (36519.0f / 878810) * z,
			(705.0f / 12673) * x - (2585.0f / 12673) * y + (705.0f / 667) * z};
	}
};

inline constexpr float lab_delta = 6.0f / 29.0f;
inline constexpr float d65_x = 0.3127f / 0.3290f;
inline constexpr float d65_z = (1.0f - 0.3127f - 0.3290f) / 0.3290f;

// Normalize to D65 white and apply the CIE Lab piecewise response.
struct xyz_to_lab_pixel {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aXyz) const
	{
		const auto x = response(aXyz[0] / d65_x);
		const auto y = response(aXyz[1]);
		const auto z = response(aXyz[2] / d65_z);
		return math::vec3f{116.0f * y - 16.0f, 500.0f * (x - y), 200.0f * (y - z)};
	}

	IANE_DECL_HYBRID static float response(float aValue)
	{
		return aValue > lab_delta * lab_delta * lab_delta ? std::cbrt(aValue) :
			aValue / (3.0f * lab_delta * lab_delta) + 4.0f / 29.0f;
	}
};

// Invert the CIE Lab response and restore the D65 reference white.
struct lab_to_xyz_pixel {
	IANE_DECL_HYBRID math::vec3f operator()(math::vec3f aLab) const
	{
		const float y = (aLab[0] + 16.0f) / 116.0f;
		return math::vec3f{d65_x * response(y + aLab[1] / 500.0f), response(y),
			d65_z * response(y - aLab[2] / 200.0f)};
	}

	IANE_DECL_HYBRID static float response(float aValue)
	{
		return aValue > lab_delta ? aValue * aValue * aValue :
			3.0f * lab_delta * lab_delta * (aValue - 4.0f / 29.0f);
	}
};

} // namespace detail

/// Convert named RGB byte channels to float channels on the unit scale.
inline constexpr detail::normalize_rgb8 rgb8_to_unit{};
/// Decode finite sRGB channels to linear sRGB without clipping.
inline constexpr detail::decode_srgb srgb_to_linear{};
/// Encode finite linear sRGB channels to sRGB without clipping.
inline constexpr detail::encode_srgb linear_to_srgb{};
/// Convert unit-range RGB to HSV with hue in turns and zero hue for gray.
inline constexpr detail::rgb_to_hsv_pixel rgb_to_hsv{};
/// Convert HSV to unit-range RGB. Hue is finite and wraps, S and V are in [0, 1].
inline constexpr detail::hsv_to_rgb_pixel hsv_to_rgb{};
/// Return relative luminance Y from linear sRGB.
inline constexpr detail::linear_luminance linear_rgb_to_luminance{};
/// Convert linear sRGB to D65 XYZ with reference white Y = 1.
inline constexpr detail::linear_rgb_to_xyz_pixel linear_rgb_to_xyz{};
/// Convert D65 XYZ to linear sRGB without gamut clipping.
inline constexpr detail::xyz_to_linear_rgb_pixel xyz_to_linear_rgb{};
/// Convert D65 XYZ to CIE Lab in conventional L*, a*, b* units.
inline constexpr detail::xyz_to_lab_pixel xyz_to_lab{};
/// Convert CIE Lab referenced to D65 to XYZ with reference white Y = 1.
inline constexpr detail::lab_to_xyz_pixel lab_to_xyz{};

} // namespace iane
