#pragma once

#include "io.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace io {

/// Read a positive odd-sized convolution kernel with finite coefficients.
inline iane::host_image<float, 2> read_kernel(const std::string& aPath)
{
	std::ifstream file(aPath);
	int width, height;
	if (!(file >> width >> height) || width <= 0 || height <= 0 ||
	    width % 2 == 0 || height % 2 == 0 ||
	    width > std::numeric_limits<int>::max() / static_cast<int>(sizeof(float))) {
		throw std::invalid_argument("Expected positive odd kernel dimensions in " + aPath);
	}

	iane::host_image<float, 2> kernel{width, height};
	auto coefficients = kernel.view();
	for (int y = 0; y < height; ++y) {
		for (int x = 0; x < width; ++x) {
			auto& value = coefficients[decltype(coefficients)::coord_t{x, y}];
			if (!(file >> value) || !std::isfinite(value)) {
				throw std::invalid_argument("Expected finite kernel coefficients in " + aPath);
			}
		}
	}
	if (!(file >> std::ws).eof()) {
		throw std::invalid_argument("Unexpected data after kernel coefficients in " + aPath);
	}
	return kernel;
}

} // namespace
