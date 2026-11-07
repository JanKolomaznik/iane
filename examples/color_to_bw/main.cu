// API design example. The algorithm and execution headers are planned.
#include <iane/algorithms/copy.hpp>
#include <iane/algorithms/grayscale.hpp>
#include <iane/execution/run_on.hpp>

#include "../common/io.hpp"

#include <boost/program_options.hpp>

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace po = boost::program_options;

namespace {

iane::rgb_weights parse_weights(const std::string& text)
{
	std::istringstream input(text);
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

int main(int argc, char** argv)
{
	try {
		po::options_description options("color_to_bw INPUT OUTPUT [options]");
		options.add_options()
			("help,h", "Show help")
			("input", po::value<std::string>()->required(), "RGB8 PNG path")
			("output", po::value<std::string>()->required(), "Grayscale PNG path")
			("backend", po::value<std::string>()->default_value("cpu"), "cpu or cuda")
			("device", po::value<int>()->default_value(0), "CUDA device index")
			("weights", po::value<std::string>()->default_value("0.2126,0.7152,0.0722"),
				"Comma-separated R,G,B weights")
			("warmup", po::value<int>()->default_value(2), "Number of warm-up runs")
			("runs", po::value<int>()->default_value(10), "Number of measured runs");

		po::positional_options_description paths;
		paths.add("input", 1).add("output", 1);
		po::variables_map args;
		po::store(po::command_line_parser(argc, argv).options(options).positional(paths).run(), args);
		if (args.count("help")) {
			std::cout << options << '\n';
			return 0;
		}
		po::notify(args);

		const auto& backend = args["backend"].as<std::string>();
		const int device = args["device"].as<int>();
		const int warmup = args["warmup"].as<int>();
		const int runs = args["runs"].as<int>();
		if (backend != "cpu" && backend != "cuda") {
			throw std::invalid_argument("Backend must be cpu or cuda");
		}
		if (device < 0 || warmup < 0 || runs <= 0) {
			throw std::invalid_argument("Device and warmup must be nonnegative and runs must be positive");
		}
		const auto weights = parse_weights(args["weights"].as<std::string>());
		const iane::run_options execution{
			.backend = backend == "cpu" ? iane::backend::cpu : iane::backend::cuda,
			.device = device,
			.warmup = warmup,
			.runs = runs,
		};

		auto input = io::read_rgb8(args["input"].as<std::string>());
		iane::host_image<std::uint8_t, 2> output{input.dimensions()};

		iane::run_on(execution, input.const_view(), output.view(),
			[&](auto src, auto dst, auto& exec) {
				return iane::copy(iane::grayscale(src, weights), dst, exec);
			});

		io::write_gray8(args["output"].as<std::string>(), output);
	} catch (const std::exception& error) {
		std::cerr << "color_to_bw: " << error.what() << '\n';
		return 1;
	}
}
