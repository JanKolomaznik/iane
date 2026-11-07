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

namespace {

template<class TPixel>
iane::host_image<TPixel, 2> read_png(const std::string& aPath, int aColorType, int aComponents)
{
	const auto close_file = [](std::FILE* aFile) { std::fclose(aFile); };
	const std::unique_ptr<std::FILE, decltype(close_file)> file{
		std::fopen(aPath.c_str(), "rb"), close_file};
	if (!file) {
		throw std::runtime_error(aPath + ": unable to open PNG");
	}

	// The PNG signature and IHDR identify the original pixel format.
	constexpr unsigned char png_header[] = {
		137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 'I', 'H', 'D', 'R'
	};
	unsigned char header[26];
	if (std::fread(header, 1, sizeof(header), file.get()) != sizeof(header)
	    || std::memcmp(header, png_header, sizeof(png_header)) != 0
	    || header[24] != 8 || header[25] != aColorType) {
		throw std::runtime_error(aPath + (aColorType == 0 ? ": expected a gray8 PNG" : ": expected an RGB8 PNG"));
	}
	std::rewind(file.get());

	int width, height, channels;
	const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{
		stbi_load_from_file(file.get(), &width, &height, &channels, aComponents),
		stbi_image_free};
	if (!pixels) {
		throw std::runtime_error(aPath + ": unable to decode PNG");
	}
	if (width > std::numeric_limits<int>::max() / static_cast<int>(sizeof(TPixel))) {
		throw std::runtime_error(aPath + ": PNG width exceeds IANE stride limits");
	}

	iane::host_image<TPixel, 2> result{width, height};
	std::memcpy(result.pointer(), pixels.get(),
		static_cast<std::size_t>(result.strides()[1]) * height);
	return result;
}

} // namespace

iane::host_image<rgb8_pixel, 2> read_rgb8(const std::string& aPath)
{
	return read_png<rgb8_pixel>(aPath, 2, STBI_rgb);
}

iane::host_image<std::uint8_t, 2> read_gray8(const std::string& aPath)
{
	return read_png<std::uint8_t>(aPath, 0, STBI_grey);
}

void write_gray8(const std::string& aPath, const iane::host_image<std::uint8_t, 2>& aInput)
{
	const auto size = aInput.dimensions();
	if (!stbi_write_png(aPath.c_str(), size[0], size[1], 1, aInput.pointer(), aInput.strides()[1])) {
		throw std::runtime_error(aPath + ": unable to write PNG");
	}
}

} // namespace io
