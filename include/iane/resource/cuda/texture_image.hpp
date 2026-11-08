#pragma once

#include <iane/concepts/view.hpp>
#include <iane/execution/context.hpp>
#include <iane/math/index.hpp>
#include <iane/view/texture_image_view.hpp>

#if defined(__CUDACC__)
#include <memory>

namespace iane {

/**
 * @brief Move-only owner of a 1D, 2D, or 3D CUDA array and its read-only texture object.
 *
 * Keep the owner alive on its CUDA device until all uploads and reads finish.
 * Upload with copy and pass const_view() to CUDA algorithms.
 */
template<texture_element TElement, int tDim = 2>
requires (tDim >= 1 && tDim <= 3)
class texture_image {
public:
	using extents_t = typename math::dim_traits<tDim>::extents_t;
	using const_view_t = texture_image_view<TElement, tDim>;
	static constexpr int dimension = tDim;

private:
	// Release the texture object before the underlying CUDA array.
	struct allocation {
		explicit allocation(extents_t aSize) : mSize(aSize) {}
		~allocation()
		{
			if (mTexture) {
				cudaDestroyTextureObject(mTexture);
			}
			if (mArray) {
				cudaFreeArray(mArray);
			}
		}

		extents_t mSize;
		cudaArray_t mArray{};
		cudaTextureObject_t mTexture{};
	};

public:
	/**
	 * @brief Allocate a texture on the current CUDA device.
	 * @param aSize Logical extent in texels. An empty extent requires no CUDA allocation.
	 * @throws std::invalid_argument For negative extents.
	 * @throws std::runtime_error If CUDA allocation or texture creation fails.
	 */
	explicit texture_image(extents_t aSize) : mStorage(std::make_unique<allocation>(aSize))
	{
		if (math::detail::element_count<tDim>(aSize) == 0) {
			return;
		}
		const auto channel = cudaCreateChannelDesc<TElement>();
		if constexpr (tDim == 1) {
			detail::check_cuda(cudaMallocArray(&mStorage->mArray, &channel, aSize),
				"CUDA texture array allocation");
		} else if constexpr (tDim == 2) {
			detail::check_cuda(cudaMallocArray(&mStorage->mArray, &channel, aSize[0], aSize[1]),
				"CUDA texture array allocation");
		} else {
			detail::check_cuda(cudaMalloc3DArray(&mStorage->mArray, &channel,
				make_cudaExtent(aSize[0], aSize[1], aSize[2])), "CUDA texture array allocation");
		}

		cudaResourceDesc resource{};
		resource.resType = cudaResourceTypeArray;
		resource.res.array.array = mStorage->mArray;

		cudaTextureDesc sampler{};
		for (int axis = 0; axis < tDim; ++axis) {
			sampler.addressMode[axis] = cudaAddressModeClamp;
		}
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
 * @brief Upload a matching host or CUDA memory view into a texture image.
 * @param aSource Source with contiguous elements within each row and optional row and slice padding.
 * @param aDestination Texture on the CUDA device used for the upload.
 * @param aPolicy Stream used for the transfer.
 * @return Completion handle. Keep both allocations alive until completion.
 * @throws std::invalid_argument For mismatched extents or invalid strides.
 * @throws std::runtime_error If CUDA reports an upload error.
 */
template<concepts::MemoryView TSource, texture_element TElement, int tDim>
requires (TSource::dimension == tDim && std::same_as<typename TSource::value_type, TElement>)
cuda_execution copy(TSource aSource, texture_image<TElement, tDim>& aDestination, cuda_policy aPolicy = {})
{
	using math::detail::axis;
	const auto size = aDestination.dimensions();
	const auto source_size = aSource.dimensions();
	for (int i = 0; i < tDim; ++i) {
		if (axis(source_size, i) != axis(size, i)) {
			throw std::invalid_argument("CUDA texture upload requires matching dimensions");
		}
	}
	if (math::detail::element_count<tDim>(size) == 0) {
		return {aPolicy.stream};
	}

	const auto strides = aSource.strides();
	const auto width = static_cast<std::size_t>(axis(size, 0));
	const auto row_bytes = width * sizeof(TElement);
	if (axis(strides, 0) != sizeof(TElement)) {
		throw std::invalid_argument("CUDA texture upload requires contiguous rows");
	}
	constexpr auto direction = concepts::HostView<TSource> ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
	if constexpr (tDim == 1) {
		cudaMemcpy3DParms transfer{};
		transfer.srcPtr = make_cudaPitchedPtr(const_cast<TElement*>(aSource.pointer()), row_bytes, row_bytes, 1);
		transfer.dstArray = aDestination.array();
		transfer.extent = make_cudaExtent(width, 1, 1);
		transfer.kind = direction;
		detail::check_cuda(cudaMemcpy3DAsync(&transfer, aPolicy.stream), "CUDA texture upload");
	} else {
		const auto pitch = static_cast<std::size_t>(strides[1]);
		if (strides[1] < 0 || pitch < row_bytes) {
			throw std::invalid_argument("CUDA texture upload requires a valid row pitch");
		}
		if constexpr (tDim == 2) {
			detail::check_cuda(cudaMemcpy2DToArrayAsync(aDestination.array(), 0, 0, aSource.pointer(),
				pitch, row_bytes, size[1], direction, aPolicy.stream), "CUDA texture upload");
		} else {
			const auto height = static_cast<std::size_t>(size[1]);
			const auto depth = static_cast<std::size_t>(size[2]);
			const auto slice = static_cast<std::size_t>(strides[2]);
			// Check the occupied rows without multiplying pitch by height.
			if (strides[2] < 0 || slice < row_bytes || (slice - row_bytes) / pitch < height - 1) {
				throw std::invalid_argument("CUDA texture upload requires a valid slice stride");
			}
			cudaMemcpy3DParms transfer{};
			transfer.dstArray = aDestination.array();
			transfer.kind = direction;
			auto* source = const_cast<TElement*>(aSource.pointer());
			if (slice % pitch == 0 && slice / pitch >= height) {
				transfer.srcPtr = make_cudaPitchedPtr(source, pitch, row_bytes, slice / pitch);
				transfer.extent = make_cudaExtent(width, height, depth);
				detail::check_cuda(cudaMemcpy3DAsync(&transfer, aPolicy.stream), "CUDA texture upload");
			} else {
				// CUDA describes slice spacing in whole pitched rows. Copy each slice when needed.
				transfer.extent = make_cudaExtent(width, height, 1);
				auto* bytes = reinterpret_cast<char*>(source);
				for (std::size_t z = 0; z < depth; ++z) {
					transfer.srcPtr = make_cudaPitchedPtr(bytes + z * slice, pitch, row_bytes, height);
					transfer.dstPos.z = z;
					detail::check_cuda(cudaMemcpy3DAsync(&transfer, aPolicy.stream), "CUDA texture slice upload");
				}
			}
		}
	}
	return {aPolicy.stream};
}

} // namespace iane
#endif
