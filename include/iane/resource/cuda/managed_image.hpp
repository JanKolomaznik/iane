#pragma once

#include <iane/execution/context.hpp>
#include <iane/image/image.hpp>
#include <iane/view/make_view.hpp>

#if defined(__CUDACC__)
#include <memory>

namespace iane::detail {

// Own one packed managed allocation and provide its host view representation.
template<class TElement, int tDim>
struct managed_image_resource {
	using extents_t = typename math::dim_traits<tDim>::extents_t;
	using view_t = host_image_view<TElement, tDim>;
	using const_view_t = const_host_image_view<TElement, tDim>;

	// Release the managed allocation using its CUDA allocator.
	struct deleter {
		void operator()(TElement* aPointer) const noexcept { cudaFree(aPointer); }
	};
	using owner_t = std::unique_ptr<TElement, deleter>;

	static owner_t allocate(const extents_t& aSize)
	{
		const auto count = resource::detail::packed_layout<TElement, tDim>(aSize).first;
		if (count == 0) {
			return {};
		}
		TElement* data = nullptr;
		check_cuda(cudaMallocManaged(&data, count * sizeof(TElement)), "CUDA managed image allocation");
		return owner_t{data};
	}

	static TElement* pointer(const owner_t& aOwner) { return aOwner.get(); }

	static extents_t strides(const extents_t& aSize, const owner_t&)
	{
		return resource::detail::packed_layout<TElement, tDim>(aSize).second;
	}
};

} // namespace iane::detail

namespace iane {

/**
 * @brief Move-only CUDA managed allocation with explicit host and device views.
 * @tparam TElement Trivially copyable element type.
 * @tparam tDim Positive image rank.
 *
 * Both view types borrow the same packed storage. Selecting a view performs
 * no transfer or synchronization. The caller orders host and GPU access and
 * keeps the allocation alive and unchanged through queued work.
 */
template<class TElement, int tDim = 2>
requires (tDim > 0 && std::is_trivially_copyable_v<TElement>)
class managed_image {
	using storage_t = image<TElement, tDim, detail::managed_image_resource<TElement, tDim>>;

public:
	using extents_t = typename storage_t::extents_t;

	/**
	 * @brief Allocate packed managed storage for the requested extents.
	 * @param aSize Nonnegative extents. An empty extent allocates no storage.
	 * @throws std::invalid_argument If an extent is negative.
	 * @throws std::overflow_error If the packed layout is not representable.
	 * @throws std::runtime_error If the CUDA allocation fails.
	 */
	explicit managed_image(extents_t aSize) : mStorage(aSize) {}

	/// Return the logical extents.
	extents_t dimensions() const { return mStorage.dimensions(); }

	/// Borrow a writable CPU view.
	auto host_view() & { return mStorage.view(); }
	/// Borrow a read-only CPU view.
	auto host_view() const & { return mStorage.const_view(); }
	auto host_view() const && = delete;

	/// Borrow a writable CUDA view.
	auto device_view() &
	{
		return device_image_view<TElement, tDim>{mStorage.pointer(), dimensions(), mStorage.strides()};
	}

	/// Borrow a read-only CUDA view.
	auto device_view() const &
	{
		return const_device_image_view<TElement, tDim>{mStorage.pointer(), dimensions(), mStorage.strides()};
	}
	auto device_view() const && = delete;

private:
	storage_t mStorage;
};

} // namespace iane
#endif
