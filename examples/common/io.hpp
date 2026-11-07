#pragma once

#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>

#include <cstdint>
#include <string>

namespace io {

/// Packed RGB channels in the order returned by stb_image.
struct rgb8_pixel {
	std::uint8_t r;
	std::uint8_t g;
	std::uint8_t b;
};

static_assert(sizeof(rgb8_pixel) == 3);

/// Decode an RGB8 PNG into an owning IANE host image.
iane::host_image<rgb8_pixel, 2> read_rgb8(const std::string& path);

/// Encode a packed IANE host image as an 8-bit grayscale PNG.
void write_gray8(const std::string& path, const iane::host_image<std::uint8_t, 2>& input);

} // namespace io
