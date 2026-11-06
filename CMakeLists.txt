cmake_minimum_required(VERSION 3.20)

# ------------------------------------------------------------
# Project configuration
# ------------------------------------------------------------
project(IANE
	LANGUAGES CXX CUDA
)

# Default to C++20 for host and device code
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable extended lambdas in CUDA (optional but useful)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")

# ------------------------------------------------------------
# IANE header-only interface library
# ------------------------------------------------------------

add_library(iane INTERFACE)

target_include_directories(iane INTERFACE
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# CUDA device code must be enabled even for header-only declarations
target_compile_features(iane INTERFACE cxx_std_20)

# Let CUDA propagate host/device flags correctly for INTERFACE targets
set_target_properties(iane PROPERTIES
	INTERFACE_COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>"
)

# ------------------------------------------------------------
# Options for building tests and tools
# ------------------------------------------------------------
option(IANE_BUILD_TESTS "Build IANE unit tests" ON)
option(IANE_BUILD_EXAMPLES "Build example programs" OFF)

# ------------------------------------------------------------
# Unit tests
# ------------------------------------------------------------
if(IANE_BUILD_TESTS)
	enable_testing()

	find_package(Boost CONFIG REQUIRED COMPONENTS unit_test_framework)

	add_subdirectory(test)
endif()

# ------------------------------------------------------------
# Examples or CLI tools (future expansion)
# ------------------------------------------------------------
if(IANE_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()
