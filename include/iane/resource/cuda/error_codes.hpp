#pragma once

/**
 * @file error_codes.hpp
 * @brief CUDA error name mapping using X-macro table (explicit & duplication-free).
 *
 * This header defines a single source of truth listing all CUDA error enums.
 * From this list, constexpr tables and utilities are automatically generated.
 *
 * This approach preserves the explicit pairing between enum values and their
 * names (as in the original CUGIP implementation), while eliminating the
 * need to repeat enum values twice.
 *
 * @ingroup iane_resource_cuda
 */

#include <string_view>
#include <iostream>
#include <iane/resource/cuda.hpp>

namespace iane::cuda {

/**
 * @brief X-macro master list of CUDA error enums.
 *
 * Every entry X(name) expands into multiple constructs:
 * 	- a constexpr entry in the lookup table
 * 	- a print entry
 * 	- any future extensions
 *
 * This ensures the enum name appears **exactly once**.
 */
#define IANE_CUDA_ERROR_LIST(X) \
	X(cudaSuccess) \
	X(cudaErrorMissingConfiguration) \
	X(cudaErrorMemoryAllocation) \
	X(cudaErrorInitializationError) \
	X(cudaErrorLaunchFailure) \
	X(cudaErrorPriorLaunchFailure) \
	X(cudaErrorLaunchTimeout) \
	X(cudaErrorLaunchOutOfResources) \
	X(cudaErrorInvalidDeviceFunction) \
	X(cudaErrorInvalidConfiguration) \
	X(cudaErrorInvalidDevice) \
	X(cudaErrorInvalidValue) \
	X(cudaErrorInvalidPitchValue) \
	X(cudaErrorInvalidSymbol) \
	X(cudaErrorMapBufferObjectFailed) \
	X(cudaErrorUnmapBufferObjectFailed) \
	X(cudaErrorInvalidHostPointer) \
	X(cudaErrorInvalidDevicePointer) \
	X(cudaErrorInvalidTexture) \
	X(cudaErrorInvalidTextureBinding) \
	X(cudaErrorInvalidChannelDescriptor) \
	X(cudaErrorInvalidMemcpyDirection) \
	X(cudaErrorAddressOfConstant) \
	X(cudaErrorTextureFetchFailed) \
	X(cudaErrorTextureNotBound) \
	X(cudaErrorSynchronizationError) \
	X(cudaErrorInvalidFilterSetting) \
	X(cudaErrorInvalidNormSetting) \
	X(cudaErrorMixedDeviceExecution) \
	X(cudaErrorCudartUnloading) \
	X(cudaErrorUnknown) \
	X(cudaErrorNotYetImplemented) \
	X(cudaErrorMemoryValueTooLarge) \
	X(cudaErrorInvalidResourceHandle) \
	X(cudaErrorNotReady) \
	X(cudaErrorInsufficientDriver) \
	X(cudaErrorSetOnActiveProcess) \
	X(cudaErrorInvalidSurface) \
	X(cudaErrorNoDevice) \
	X(cudaErrorECCUncorrectable) \
	X(cudaErrorSharedObjectSymbolNotFound) \
	X(cudaErrorSharedObjectInitFailed) \
	X(cudaErrorUnsupportedLimit) \
	X(cudaErrorDuplicateVariableName) \
	X(cudaErrorDuplicateTextureName) \
	X(cudaErrorDuplicateSurfaceName) \
	X(cudaErrorDevicesUnavailable) \
	X(cudaErrorInvalidKernelImage) \
	X(cudaErrorNoKernelImageForDevice) \
	X(cudaErrorIncompatibleDriverContext) \
	X(cudaErrorPeerAccessAlreadyEnabled) \
	X(cudaErrorPeerAccessNotEnabled) \
	X(cudaErrorDeviceAlreadyInUse) \
	X(cudaErrorProfilerDisabled) \
	X(cudaErrorProfilerNotInitialized) \
	X(cudaErrorProfilerAlreadyStarted) \
	X(cudaErrorProfilerAlreadyStopped) \
	X(cudaErrorAssert) \
	X(cudaErrorTooManyPeers) \
	X(cudaErrorHostMemoryAlreadyRegistered) \
	X(cudaErrorHostMemoryNotRegistered) \
	X(cudaErrorOperatingSystem) \
	X(cudaErrorStartupFailure) \
	X(cudaErrorApiFailureBase)

/**
 * @brief One entry in the compile-time error table.
 */
struct cuda_error_entry {
	cudaError_t	code;
	std::string_view	name;
};

/**
 * @brief Generate the constexpr lookup table using the X-macro list.
 */
inline constexpr cuda_error_entry cuda_error_table[] = {
#define IANE_CUDA_ERR_ITEM(NAME) { NAME, #NAME },
	IANE_CUDA_ERROR_LIST(IANE_CUDA_ERR_ITEM)
#undef IANE_CUDA_ERR_ITEM
};

/**
 * @brief Lookup: name → string (linear scan; table is tiny).
 */
inline constexpr std::string_view
get_error_name(cudaError_t aErr)
{
	for (const auto &entry : cuda_error_table) {
		if (entry.code == aErr) {
			return entry.name;
		}
	}
	return {};
}

/**
 * @brief Debug print of all CUDA error mappings.
 */
inline void
print_all_error_names()
{
#define IANE_CUDA_PRINT_NAME(NAME) \
	std::cout << "(" << int(NAME) << "): " << #NAME << '\n';

	IANE_CUDA_ERROR_LIST(IANE_CUDA_PRINT_NAME)

#undef IANE_CUDA_PRINT_NAME
}

} // namespace iane::cuda
