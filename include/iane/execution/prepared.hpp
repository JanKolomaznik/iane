#pragma once

#include <iane/execution/context.hpp>
#include <iane/resource/packed_layout.hpp>
#include <iane/view/make_view.hpp>

#include <concepts>
#include <memory>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace iane {
namespace detail {

// Carve typed or raw temporary regions out of the context's current allocation.
template<class TPolicy>
class scratch_space {
public:
	scratch_space(void* aData, std::size_t aSize) : mData(aData), mSize(aSize) {}

	void* bytes(std::size_t aSize, std::size_t aAlignment = alignof(std::max_align_t))
	{
		if (aSize == 0) {
			return nullptr;
		}
		if (aAlignment == 0 || aAlignment > alignof(std::max_align_t) ||
		    (aAlignment & (aAlignment - 1)) != 0) {
			throw std::invalid_argument("Unsupported scratch alignment");
		}
		if (aSize > mSize || !std::align(aAlignment, aSize, mData, mSize)) {
			throw std::length_error("Prepared operation exceeded its scratch reservation");
		}
		auto* result = static_cast<std::byte*>(mData);
		mData = result + aSize;
		mSize -= aSize;
		return result;
	}

	template<class TElement, int tDim>
	auto image(const typename math::dim_traits<tDim>::extents_t& aSize)
	{
		static_assert(std::is_trivially_copyable_v<TElement>);
		const auto count = resource::detail::packed_layout<TElement, tDim>(aSize).first;
		auto* pointer = static_cast<TElement*>(bytes(count * sizeof(TElement), alignof(TElement)));
		if constexpr (std::same_as<TPolicy, cpu_policy>) {
			return host::make_view<TElement, tDim>(pointer, aSize);
		} else {
			return device::make_view<TElement, tDim>(pointer, aSize);
		}
	}

private:
	void* mData;
	std::size_t mSize;
};

// Release the workspace with the allocator used by the context's backend.
template<class TPolicy>
struct scratch_deleter {
	void operator()(void* aPointer) const noexcept
	{
#if defined(__CUDACC__)
		if constexpr (std::same_as<TPolicy, cuda_policy>) {
			cudaFree(aPointer);
		} else
#endif
		{
			::operator delete(aPointer);
		}
	}
};

} // namespace detail

/**
 * @brief Prepare and run algorithms with one shared scratch allocation.
 *
 * Calls on a context must be serialized by the caller. CUDA contexts borrow
 * a stream and must be used and destroyed on its device. The stream must
 * outlive the context. Keep input and output storage alive through completion.
 */
template<class TPolicy>
class execution_context {
	// Tie an algorithm's prepared state to the context that reserved its scratch.
	template<class TState>
	class prepared_operation {
		friend class execution_context;

		prepared_operation(const execution_context* aOwner, TState&& aState)
			: mOwner(aOwner), mState(std::move(aState)) {}

		const execution_context* mOwner;
		TState mState;
	};

public:
	/// Create an empty context using the supplied backend policy.
	explicit execution_context(TPolicy aPolicy) : mPolicy(aPolicy) {}
	execution_context(const execution_context&) = delete;
	execution_context& operator=(const execution_context&) = delete;

	/// Finish outstanding CUDA work before freeing the shared allocation.
	~execution_context()
	{
#if defined(__CUDACC__)
		if constexpr (std::same_as<TPolicy, cuda_policy>) {
			if (mSubmitted) {
				cudaStreamSynchronize(mPolicy.stream);
			}
		}
#endif
	}

	/**
	 * @brief Prepare an algorithm from prototype views and reserve its scratch.
	 * @return Prepared operation bound to this context.
	 */
	template<class TAlgorithm, class TSrc, class TDst>
	auto prepare(TAlgorithm aAlgorithm, TSrc aSource, TDst aDestination)
	{
		auto state = aAlgorithm.prepare(aSource, aDestination, mPolicy);
		reserve(state.scratch_bytes());
		return prepared_operation<decltype(state)>{this, std::move(state)};
	}

	/**
	 * @brief Run a prepared operation on compatible views.
	 * @return Backend completion handle.
	 * @throws std::invalid_argument If the operation belongs to another context.
	 */
	template<class TState, class TSrc, class TDst>
	auto run(prepared_operation<TState>& aOperation, TSrc aSource, TDst aDestination)
	{
		if (aOperation.mOwner != this) {
			throw std::invalid_argument("Prepared operation belongs to a different execution context");
		}
#if defined(__CUDACC__)
		if constexpr (std::same_as<TPolicy, cuda_policy>) {
			mSubmitted = true;
		}
#endif
		detail::scratch_space<TPolicy> scratch(mScratch.get(), mCapacity);
		return aOperation.mState.run(aSource, aDestination, scratch, mPolicy);
	}

	/// Return the current shared scratch capacity in bytes.
	std::size_t scratch_capacity() const noexcept { return mCapacity; }

private:
	void reserve(std::size_t aSize)
	{
		if (aSize <= mCapacity) {
			return;
		}
		void* storage = nullptr;
#if defined(__CUDACC__)
		if constexpr (std::same_as<TPolicy, cuda_policy>) {
			if (mSubmitted) {
				cuda_execution{mPolicy.stream}.wait();
				mSubmitted = false;
			}
			detail::check_cuda(cudaMalloc(&storage, aSize), "Allocate CUDA scratch");
		} else
#endif
		{
			storage = ::operator new(aSize);
		}
		mScratch.reset(storage);
		mCapacity = aSize;
	}

	TPolicy mPolicy;
	std::unique_ptr<void, detail::scratch_deleter<TPolicy>> mScratch;
	std::size_t mCapacity{};
#if defined(__CUDACC__)
	bool mSubmitted{};
#endif
};

/// Create a synchronous CPU context, serial by default.
inline auto make_execution_context(cpu_policy aPolicy = {})
{
	return execution_context<cpu_policy>(aPolicy);
}

#if defined(__CUDACC__)
/// Create a CUDA context borrowing the policy's stream.
inline auto make_execution_context(cuda_policy aPolicy)
{
	return execution_context<cuda_policy>(aPolicy);
}
#endif

} // namespace iane
