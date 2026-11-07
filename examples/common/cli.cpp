#include "cli.hpp"

#include <boost/json.hpp>

#include <fstream>
#include <iterator>
#include <vector>

namespace cli {
namespace detail {
namespace {

std::string option_token(const boost::json::value& aValue)
{
	if (aValue.is_string()) {
		const auto& value = aValue.as_string();
		return {value.data(), value.size()};
	}
	if (aValue.is_number() || aValue.is_bool()) {
		return boost::json::serialize(aValue);
	}
	throw std::invalid_argument("Expected a string, number or boolean");
}

} // namespace

po::parsed_options parse_command_line(int aArgc, char** aArgv,
	const po::options_description& aOptions, const po::positional_options_description& aPaths)
{
	std::vector<std::string> arguments(aArgv + 1, aArgv + aArgc);
	for (auto& argument : arguments) {
		if (argument == "--") {
			break;
		}
		if (argument == "-h") {
			argument = "--help";
		}
	}
	// A minus sign on a numeric value must not start a short option.
	namespace style = po::command_line_style;
	return po::command_line_parser(arguments).options(aOptions).positional(aPaths)
		.style(style::default_style & ~style::allow_short).run();
}

void store_config(const std::string& aPath, const po::options_description& aOptions,
	po::variables_map& aArgs)
{
	std::ifstream file(aPath);
	if (!file) {
		throw std::runtime_error(aPath + ": unable to open configuration");
	}
	try {
		const std::string contents{std::istreambuf_iterator<char>(file), {}};
		const auto document = boost::json::parse(contents);
		if (!document.is_object()) {
			throw std::invalid_argument("Expected a JSON object");
		}

		po::parsed_options parsed(&aOptions);
		for (const auto& entry : document.as_object()) {
			const std::string key(entry.key().data(), entry.key().size());
			const auto* description = aOptions.find_nothrow(key, false);
			if (!description || key == "config" || key == "help") {
				throw std::invalid_argument("Unknown configuration option '" + key + "'");
			}
			if (const auto found = aArgs.find(key); found != aArgs.end() && !found->second.defaulted()) {
				continue;
			}

			std::vector<std::string> tokens;
			try {
				if (entry.value().is_array()) {
					for (const auto& item : entry.value().as_array()) {
						tokens.push_back(option_token(item));
					}
				} else {
					tokens.push_back(option_token(entry.value()));
				}
				if (tokens.size() < description->semantic()->min_tokens() ||
				    tokens.size() > description->semantic()->max_tokens()) {
					throw std::invalid_argument("Wrong number of values");
				}
			} catch (const std::exception& error) {
				throw std::invalid_argument("Option '" + key + "': " + error.what());
			}
			parsed.options.emplace_back(key, tokens);
		}
		po::store(parsed, aArgs);
	} catch (const std::exception& error) {
		throw std::runtime_error(aPath + ": " + error.what());
	}
}

} // namespace detail
} // namespace cli
