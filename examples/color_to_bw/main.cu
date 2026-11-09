#include <iane/algorithms/copy.hpp>
#include <iane/algorithms/color/grayscale.hpp>

#include "../common/cli.hpp"
#include "../common/io.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

namespace po = cli::po;

namespace {

iane::rgb_weights parse_weights(const std::string& aText)
{
	std::istringstream input(aText);
	double red, green, blue;
	char first_separator, second_separator;
	if (!(input >> red >> first_separator >> green >> second_separator >> blue)
	    || first_separator != ',' || second_separator != ','
	    || !(input >> std::ws).eof()
	    || !std::isfinite(red) || !std::isfinite(green) || !std::isfinite(blue)) {
		throw std::invalid_argument("Expected three finite weights as R,G,B");
	}
	return {red, green, blue};
}

} // namespace

int main(int aArgc, char** aArgv)
{
	po::options_description options("Grayscale options");
	options.add_options()
		("weights", po::value<std::string>()->default_value("0.2126,0.7152,0.0722"),
			"Comma-separated R,G,B weights");

	return cli::run(aArgc, aArgv, "color_to_bw", options,
		[](const std::string& aInputPath, const std::string& aOutputPath,
		   const iane::run_options& aExecution, const po::variables_map& aArgs) {
			const auto weights = parse_weights(aArgs["weights"].as<std::string>());

			auto input = io::read_rgb8(aInputPath);
			iane::host_image<std::uint8_t, 2> output{input.dimensions()};

			iane::run_on(aExecution, input.const_view(), output.view(),
				[&](auto aSrc, auto aDst, auto& aExec) {
					return iane::copy(iane::grayscale(aSrc, weights), aDst, aExec);
				});

			io::write_gray8(aOutputPath, output);
		});
}
