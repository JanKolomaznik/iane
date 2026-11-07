#include <iane/algorithms/convolution.hpp>

#include "../common/cli.hpp"
#include "../common/io.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace po = cli::po;

namespace {

iane::host_image<float, 2> read_kernel(const std::string& aPath)
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

int main(int aArgc, char** aArgv)
{
	po::options_description options("Convolution options");
	options.add_options()
		("kernel", po::value<std::string>()->required(), "Text file containing kernel coefficients")
		("roi", po::value<std::vector<int>>()->multitoken(), "Crop as X Y WIDTH HEIGHT")
		("border", po::value<std::string>()->default_value("clamp"), "clamp or zero");

	return cli::run(aArgc, aArgv, "convolution", options,
		[](const std::string& aInputPath, const std::string& aOutputPath,
		   const iane::run_options& aExecution, const po::variables_map& aArgs) {
			const auto& border = aArgs["border"].as<std::string>();
			if (border != "clamp" && border != "zero") {
				throw std::invalid_argument("Border must be clamp or zero");
			}

			auto input = io::read_gray8(aInputPath);
			auto kernel = read_kernel(aArgs["kernel"].as<std::string>());
			auto source = input.const_view();
			if (aArgs.count("roi")) {
				const auto& roi = aArgs["roi"].as<std::vector<int>>();
				const auto size = source.dimensions();
				if (roi.size() != 4 || roi[0] < 0 || roi[1] < 0 ||
				    roi[2] <= 0 || roi[3] <= 0 ||
				    roi[2] > size[0] - roi[0] || roi[3] > size[1] - roi[1]) {
					throw std::invalid_argument("ROI must be a nonempty rectangle inside the input");
				}
				using view_t = decltype(source);
				source = source.subview(view_t::coord_t{roi[0], roi[1]},
					view_t::extents_t{roi[2], roi[3]});
			}
			iane::host_image<std::uint8_t, 2> output{source.dimensions()};
			const iane::convolution_options convolution{
				.border = border == "clamp" ? iane::border_mode::clamp : iane::border_mode::zero,
			};

			iane::run_on(aExecution, std::tuple{source, kernel.const_view()}, output.view(),
				[&](auto aSrc, auto aCoefficients, auto aDst, auto& aExec) {
					return iane::convolve(aSrc, aCoefficients, aDst, convolution, aExec);
				});

			io::write_gray8(aOutputPath, output);
		});
}
