#pragma once

#include <iane/execution/run_on.hpp>

#include <boost/program_options.hpp>

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace cli {

namespace po = boost::program_options;

namespace detail {

po::parsed_options parse_command_line(int aArgc, char** aArgv,
	const po::options_description& aOptions, const po::positional_options_description& aPaths);

void store_config(const std::string& aPath, const po::options_description& aOptions,
	po::variables_map& aArgs);

} // namespace detail

/**
 * @brief Parse shared options and invoke the example. CLI values override JSON.
 * @tparam tHasInput Whether the utility accepts an input image path.
 */
template<bool tHasInput = true, class TOperation>
int run(int aArgc, char** aArgv, const std::string& aName,
	const po::options_description& aOperationOptions, TOperation aOperation)
{
	try {
		po::options_description options(aName + (tHasInput ? " INPUT OUTPUT [options]" : " OUTPUT [options]"));
		options.add_options()
			("help,h", "Show help")
			("config", po::value<std::string>(), "Read options from a JSON file")
			("output", po::value<std::string>()->required(), "Output image path (.png, .jpg, .jpeg, .bmp or .tga)")
			("backend", po::value<std::string>()->default_value("cpu"), "cpu or cuda")
			("device", po::value<int>()->default_value(0), "CUDA device index")
			("warmup", po::value<int>()->default_value(2), "Number of warm-up runs")
			("runs", po::value<int>()->default_value(10), "Number of measured runs");
		po::positional_options_description paths;
		if constexpr (tHasInput) {
			options.add_options()
				("input", po::value<std::string>()->required(), "Input image path (format detected from contents)");
			paths.add("input", 1);
		}
		paths.add("output", 1);
		options.add(aOperationOptions);
		po::variables_map args;
		po::store(detail::parse_command_line(aArgc, aArgv, options, paths), args);
		if (args.count("help")) {
			std::cout << options << '\n';
			return 0;
		}
		if (args.count("config")) {
			detail::store_config(args["config"].as<std::string>(), options, args);
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
		const auto& output = args["output"].as<std::string>();
		if constexpr (tHasInput) {
			aOperation(args["input"].as<std::string>(), output, execution, args);
		} else {
			aOperation(output, execution, args);
		}
		return 0;
	} catch (const std::exception& error) {
		std::cerr << aName << ": " << error.what() << '\n';
		return 1;
	}
}

} // namespace cli
