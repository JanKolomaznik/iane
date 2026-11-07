#include "io.hpp"

#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace io {

iane::host_image<rgb8_pixel, 2> read_rgb8(const std::string& path)
{
	const auto close_file = [](std::FILE* file) { std::fclose(file); };
	const std::unique_ptr<std::FILE, decltype(close_file)> file{
		std::fopen(path.c_str(), "rb"), close_file};
	if (!file) {
		throw std::runtime_error(path + ": unable to open PNG");
	}

	// The PNG signature and IHDR identify the original pixel format.
	constexpr unsigned char png_header[] = {
		137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 'I', 'H', 'D', 'R'
	};
	unsigned char header[26];
	if (std::fread(header, 1, sizeof(header), file.get()) != sizeof(header)
	    || std::memcmp(header, png_header, sizeof(png_header)) != 0
	    || header[24] != 8 || header[25] != 2) {
		throw std::runtime_error(path + ": expected an RGB8 PNG");
	}
	std::rewind(file.get());

	int width, height, channels;
	const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{
		stbi_load_from_file(file.get(), &width, &height, &channels, STBI_rgb),
		stbi_image_free};
	if (!pixels) {
		throw std::runtime_error(path + ": unable to decode PNG");
	}
	if (width > std::numeric_limits<int>::max() / static_cast<int>(sizeof(rgb8_pixel))) {
		throw std::runtime_error(path + ": PNG width exceeds IANE stride limits");
	}

	iane::host_image<rgb8_pixel, 2> result{width, height};
	std::memcpy(result.pointer(), pixels.get(),
		static_cast<std::size_t>(result.strides()[1]) * height);
	return result;
}

void write_gray8(const std::string& path, const iane::host_image<std::uint8_t, 2>& input)
{
	const auto size = input.dimensions();
	if (!stbi_write_png(path.c_str(), size[0], size[1], 1, input.pointer(), input.strides()[1])) {
		throw std::runtime_error(path + ": unable to write PNG");
	}
}

} // namespace io
