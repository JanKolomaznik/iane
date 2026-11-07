#pragma once

#include <iane/concepts/view.hpp>
#include <iane/execution/context.hpp>
#include <iane/math/index.hpp>
#include <iane/view/texture_image_view.hpp>

#if defined(__CUDACC__)
#include <memory>

namespace iane {

/**
 * @brief Move-only owner of a 2D CUDA array and its read-only texture object.
 *
 * Keep the owner alive on its CUDA device until all uploads and reads finish.
 * Upload with copy and pass const_view() to CUDA algorithms.
 */
template<texture_element TElement>
class texture_image {
	// Release the texture object before the underlying CUDA array.
	struct allocation {
		explicit allocation(math::vec2i aSize) : mSize(aSize) {}
		~allocation()
		{
			if (mTexture) {
				cudaDestroyTextureObject(mTexture);
			}
			if (mArray) {
				cudaFreeArray(mArray);
			}
		}

		math::vec2i mSize;
		cudaArray_t mArray{};
		cudaTextureObject_t mTexture{};
	};

public:
	using extents_t = math::vec2i;
	using const_view_t = texture_image_view<TElement>;

	/**
	 * @brief Allocate a texture on the current CUDA device.
	 * @param aSize Width and height. An empty extent requires no CUDA allocation.
	 * @throws std::invalid_argument For negative extents.
	 * @throws std::runtime_error If CUDA allocation or texture creation fails.
	 */
	explicit texture_image(extents_t aSize) : mStorage(std::make_unique<allocation>(aSize))
	{
		if (math::detail::element_count<2>(aSize) == 0) {
			return;
		}
		const auto channel = cudaCreateChannelDesc<TElement>();
		detail::check_cuda(cudaMallocArray(&mStorage->mArray, &channel, aSize[0], aSize[1]),
			"CUDA texture array allocation");

		cudaResourceDesc resource{};
		resource.resType = cudaResourceTypeArray;
		resource.res.array.array = mStorage->mArray;

		cudaTextureDesc sampler{};
		sampler.addressMode[0] = cudaAddressModeClamp;
		sampler.addressMode[1] = cudaAddressModeClamp;
		sampler.filterMode = cudaFilterModePoint;
		sampler.readMode = cudaReadModeElementType;
		detail::check_cuda(cudaCreateTextureObject(&mStorage->mTexture, &resource, &sampler, nullptr),
			"CUDA texture object creation");
	}

	/// Return the logical extent in texels.
	extents_t dimensions() const { return mStorage ? mStorage->mSize : extents_t{}; }

	/// Borrow a readable CUDA view until the allocation is released.
	const_view_t const_view() const
	{
		return {mStorage ? mStorage->mTexture : cudaTextureObject_t{}, dimensions()};
	}

	/// Borrow the CUDA array for interoperability. The image retains ownership.
	cudaArray_t array() const { return mStorage ? mStorage->mArray : nullptr; }

private:
	std::unique_ptr<allocation> mStorage;
};

/**
 * @brief Upload a matching 2D host or CUDA memory view into a texture image.
 * @param aSource Source with contiguous elements within each row.
 * @param aDestination Texture on the CUDA device used for the upload.
 * @param aPolicy Stream used for the transfer.
 * @return Completion handle. Keep both allocations alive until completion.
 * @throws std::invalid_argument For mismatched extents or invalid row strides.
 * @throws std::runtime_error If CUDA reports an upload error.
 */
template<concepts::MemoryView TSource, texture_element TElement>
requires (TSource::dimension == 2 && std::same_as<typename TSource::value_type, TElement>)
cuda_execution copy(TSource aSource, texture_image<TElement>& aDestination, cuda_policy aPolicy = {})
{
	const auto size = aDestination.dimensions();
	const auto source_size = aSource.dimensions();
	if (source_size[0] != size[0] || source_size[1] != size[1]) {
		throw std::invalid_argument("CUDA texture upload requires matching dimensions");
	}
	if (size[0] == 0 || size[1] == 0) {
		return {aPolicy.stream};
	}

	const auto strides = aSource.strides();
	const auto row_bytes = static_cast<std::size_t>(size[0]) * sizeof(TElement);
	if (strides[0] != sizeof(TElement) || strides[1] < 0 ||
	    static_cast<std::size_t>(strides[1]) < row_bytes) {
		throw std::invalid_argument("CUDA texture upload requires contiguous rows and a valid pitch");
	}
	constexpr auto direction = concepts::HostView<TSource> ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
	detail::check_cuda(cudaMemcpy2DToArrayAsync(aDestination.array(), 0, 0, aSource.pointer(),
		strides[1], row_bytes, size[1], direction, aPolicy.stream), "CUDA texture upload");
	return {aPolicy.stream};
}

} // namespace iane
#endif
