#pragma once

#include <iane/execution/run_on.hpp>

#include <boost/program_options.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cli {

namespace po = boost::program_options;

/// Parse the shared image CLI and invoke the example with its parsed options.
template<class TOperation>
int run(int aArgc, char** aArgv, const std::string& aName,
	const po::options_description& aOperationOptions, TOperation aOperation)
{
	try {
		po::options_description options(aName + " INPUT OUTPUT [options]");
		options.add_options()
			("help,h", "Show help")
			("input", po::value<std::string>()->required(), "Input PNG path")
			("output", po::value<std::string>()->required(), "Output PNG path")
			("backend", po::value<std::string>()->default_value("cpu"), "cpu or cuda")
			("device", po::value<int>()->default_value(0), "CUDA device index")
			("warmup", po::value<int>()->default_value(2), "Number of warm-up runs")
			("runs", po::value<int>()->default_value(10), "Number of measured runs");
		options.add(aOperationOptions);

		po::positional_options_description paths;
		paths.add("input", 1).add("output", 1);
		po::variables_map args;
		po::store(po::command_line_parser(aArgc, aArgv).options(options).positional(paths).run(), args);
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

		const iane::run_options execution{
			.backend = backend == "cpu" ? iane::backend::cpu : iane::backend::cuda,
			.device = device,
			.warmup = warmup,
			.runs = runs,
		};
		aOperation(args["input"].as<std::string>(), args["output"].as<std::string>(), execution, args);
		return 0;
	} catch (const std::exception& error) {
		std::cerr << aName << ": " << error.what() << '\n';
		return 1;
	}
}

} // namespace cli
