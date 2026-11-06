#pragma once
/**
 * @file exception.hpp
 * @brief Backend-independent exception types for IANE.
 */

#include <iane/config.hpp>
#include <iane/logging.hpp>

#include <boost/exception/all.hpp>
#include <exception>
#include <string>
#include <filesystem>

namespace iane {

/**
 * @brief Boost.Exception-compatible message info.
 */
using MessageInfo = boost::error_info<struct tag_message, std::string>;
using FileInfo    = boost::error_info<struct tag_filename, std::filesystem::path>;
using DirInfo     = boost::error_info<struct tag_dirname, std::filesystem::path>;

/**
 * @brief Base exception for IANE.
 */
class ExceptionBase :
	public virtual boost::exception,
	public virtual std::exception
{
public:
	const char *what() const noexcept override
	{
		if (auto *msg = boost::get_error_info<MessageInfo>(*this))
			return msg->c_str();
		return "IANE exception";
	}
};

/**
 * @brief Exception thrown when two views have non-matching sizes.
 */
class IncompatibleViewSizes : public ExceptionBase {};

/// Generic throw
#define IANE_THROW(...)	throw __VA_ARGS__

} // namespace iane

// -----------------------------------------------------------------------------
// CUDA exception extension (only included when CUDA is available)
// -----------------------------------------------------------------------------

#if defined(__CUDACC__) || defined(IANE_ENABLE_CUDA)
	#include <iane/resource/cuda/exception.hpp>
#endif
