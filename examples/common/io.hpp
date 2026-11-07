#pragma once

#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>

#include <cstdint>
#include <string>
#include <variant>

namespace io {

/// Packed RGB channels in the order returned by stb_image.
struct rgb8_pixel {
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
};

static_assert(sizeof(rgb8_pixel) == 3);

/// Decode a color image into an owning RGB8 host image.
iane::host_image<rgb8_pixel, 2> read_rgb8(const std::string& aPath);

/// Decode a grayscale image into an owning gray8 host image.
iane::host_image<std::uint8_t, 2> read_gray8(const std::string& aPath);

/// Decoded image in gray8 or RGB8 layout.
using image8 = std::variant<iane::host_image<std::uint8_t, 2>, iane::host_image<rgb8_pixel, 2>>;

/// Detect and decode an image through stb, discarding alpha if present.
image8 read_image8(const std::string& aPath);

/// Encode a packed IANE host image as grayscale, using the output filename extension.
void write_gray8(const std::string& aPath, const iane::host_image<std::uint8_t, 2>& aInput);

/// Encode a packed IANE host image as RGB, using the output filename extension.
void write_rgb8(const std::string& aPath, const iane::host_image<rgb8_pixel, 2>& aInput);

} // namespace io
