#pragma once
/**
 * @file logging.hpp
 * @brief Lightweight logging utilities for the IANE library.
 *
 * Provides stream-based and boost::format-based logging macros,
 * keeping overhead minimal in release builds while enabling
 * detailed debug output when NDEBUG is not defined.
 *
 * The loggers are overridable via IANE_*_LOGGER macros prior to
 * including this header.
 */

#include <iostream>
#include <boost/format.hpp>
#include <utility>

namespace iane::detail {

/// End of recursion for format argument expansion.
inline void format_helper(boost::format &) {}

/**
 * @brief Recursively expand boost::format arguments.
 */
template<typename T, typename... Args>
inline void format_helper(boost::format &fmt, T &&value, Args &&...args)
{
	fmt % std::forward<T>(value);
	format_helper(fmt, std::forward<Args>(args)...);
}

} // namespace iane::detail

// -----------------------------------------------------------------------------
// Logger stream configuration (overridable by user)
// -----------------------------------------------------------------------------

#ifndef IANE_DEBUG_LOGGER
#define IANE_DEBUG_LOGGER	std::cout
#endif

#ifndef IANE_TRACE_LOGGER
#define IANE_TRACE_LOGGER	std::cout
#endif

#ifndef IANE_ERROR_LOGGER
#define IANE_ERROR_LOGGER	std::cerr
#endif

#ifndef IANE_INFO_LOGGER
#define IANE_INFO_LOGGER	std::cout
#endif

// -----------------------------------------------------------------------------
// Stream-based logging macros
// -----------------------------------------------------------------------------

#ifndef NDEBUG
	/// Debug print (disabled in release)
	#define IANE_DPRINT(...) \
		do { IANE_DEBUG_LOGGER << __VA_ARGS__ << std::endl; } while(false)
#else
	#define IANE_DPRINT(...)	do {} while(false)
#endif

/// Trace logging (always enabled)
#define IANE_TPRINT(...) \
	do { IANE_TRACE_LOGGER << __VA_ARGS__ << std::endl; } while(false)

/// Error logging (always enabled)
#define IANE_EPRINT(...) \
	do { IANE_ERROR_LOGGER << __VA_ARGS__ << std::endl; } while(false)

/// Informational logging (always enabled)
#define IANE_PRINT(...) \
	do { IANE_INFO_LOGGER << __VA_ARGS__ << std::endl; } while(false)


// -----------------------------------------------------------------------------
// Format-based logging (boost::format)
// -----------------------------------------------------------------------------

/**
 * @brief Internal implementation for formatted logs.
 *
 * Prefixes output with file and line number.
 */
#define IANE_LOG_FORMAT(logger, fmt_str, ...) \
	do { \
		boost::format fmt(fmt_str); \
		::iane::detail::format_helper(fmt, ##__VA_ARGS__); \
		(logger) << __FILE__ << ":" << __LINE__ << ": " \
		         << fmt << std::endl; \
	} while(false)

#ifndef NDEBUG
	/// Debug format (disabled in release)
	#define IANE_DFORMAT(fmt_str, ...) \
		IANE_LOG_FORMAT(IANE_DEBUG_LOGGER, fmt_str, __VA_ARGS__)
#else
	#define IANE_DFORMAT(fmt_str, ...)	do {} while(false)
#endif

/// Trace format
#define IANE_TFORMAT(fmt_str, ...) \
	IANE_LOG_FORMAT(IANE_TRACE_LOGGER, fmt_str, __VA_ARGS__)

/// Error format
#define IANE_EFORMAT(fmt_str, ...) \
	IANE_LOG_FORMAT(IANE_ERROR_LOGGER, fmt_str, __VA_ARGS__)

/// Info format
#define IANE_LFORMAT(fmt_str, ...) \
	IANE_LOG_FORMAT(IANE_INFO_LOGGER, fmt_str, __VA_ARGS__)
