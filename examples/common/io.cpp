#include "io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace io {

namespace {

template<class TPixel>
iane::host_image<TPixel, 2> make_image(const stbi_uc* aPixels, int aWidth, int aHeight, int aChannels)
{
	iane::host_image<TPixel, 2> result{aWidth, aHeight};
	if (aChannels == sizeof(TPixel)) {
		std::memcpy(result.pointer(), aPixels,
			static_cast<std::size_t>(result.strides()[1]) * aHeight);
	} else {
		// stb places alpha after the gray or RGB channels.
		const auto count = static_cast<std::size_t>(aWidth) * aHeight;
		for (std::size_t i = 0; i < count; ++i) {
			std::memcpy(result.pointer() + i, aPixels + i * aChannels, sizeof(TPixel));
		}
	}
	return result;
}

template<class TPixel>
void write_image(const std::string& aPath, const iane::host_image<TPixel, 2>& aInput, int aComponents)
{
	auto extension = std::filesystem::path(aPath).extension().string();
	std::ranges::transform(extension, extension.begin(),
		[](unsigned char aCharacter) { return static_cast<char>(std::tolower(aCharacter)); });
	const auto size = aInput.dimensions();
	const auto* pixels = aInput.pointer();
	const auto* path = aPath.c_str();

	int written;
	if (extension == ".png") {
		written = stbi_write_png(path, size[0], size[1], aComponents, pixels, aInput.strides()[1]);
	} else if (extension == ".jpg" || extension == ".jpeg") {
		written = stbi_write_jpg(path, size[0], size[1], aComponents, pixels, 95);
	} else if (extension == ".bmp") {
		written = stbi_write_bmp(path, size[0], size[1], aComponents, pixels);
	} else if (extension == ".tga") {
		written = stbi_write_tga(path, size[0], size[1], aComponents, pixels);
	} else {
		throw std::invalid_argument(aPath + ": output extension must be .png, .jpg, .jpeg, .bmp or .tga");
	}
	if (!written) {
		throw std::runtime_error(aPath + ": unable to write image");
	}
}

} // namespace

image8 read_image8(const std::string& aPath)
{
	int width, height, channels;
	const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> pixels{
		stbi_load(aPath.c_str(), &width, &height, &channels, 0),
		stbi_image_free};
	if (!pixels) {
		throw std::runtime_error(aPath + ": unable to decode image");
	}
	if (channels <= 2) {
		return make_image<std::uint8_t>(pixels.get(), width, height, channels);
	}
	return make_image<rgb8_pixel>(pixels.get(), width, height, channels);
}

iane::host_image<rgb8_pixel, 2> read_rgb8(const std::string& aPath)
{
	auto image = read_image8(aPath);
	if (auto* rgb = std::get_if<iane::host_image<rgb8_pixel, 2>>(&image)) {
		return std::move(*rgb);
	}
	throw std::runtime_error(aPath + ": expected a color image");
}

iane::host_image<std::uint8_t, 2> read_gray8(const std::string& aPath)
{
	auto image = read_image8(aPath);
	if (auto* gray = std::get_if<iane::host_image<std::uint8_t, 2>>(&image)) {
		return std::move(*gray);
	}
	throw std::runtime_error(aPath + ": expected a grayscale image");
}

void write_gray8(const std::string& aPath, const iane::host_image<std::uint8_t, 2>& aInput)
{
	write_image(aPath, aInput, STBI_grey);
}

void write_rgb8(const std::string& aPath, const iane::host_image<rgb8_pixel, 2>& aInput)
{
	write_image(aPath, aInput, STBI_rgb);
}

} // namespace io
