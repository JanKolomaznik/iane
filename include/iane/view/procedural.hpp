#pragma once

#include <iane/concepts/view.hpp>
#include <iane/math/index.hpp>

#include <utility>

namespace iane {
namespace detail {

// Preserve exactly one execution tag without storing the domain view.
template<bool tHost>
struct procedural_backend {
	using is_device_view_tag = void;
};

template<>
struct procedural_backend<true> {
	using is_host_view_tag = void;
};

template<class TView>
concept procedural_domain = concepts::ViewDomain<TView> &&
	(concepts::HostView<TView> != concepts::DeviceView<TView>);

// Store the shape and coordinate callable for a lazy, read-only view.
template<procedural_domain TDomain, class TFunction>
struct procedural_view : procedural_backend<concepts::HostView<TDomain>> {
	using coord_t = typename TDomain::coord_t;
	using extents_t = typename TDomain::extents_t;
	using value_type = std::remove_cvref_t<decltype(
		std::declval<const TFunction&>()(std::declval<const coord_t&>()))>;
	static constexpr int dimension = TDomain::dimension;

	extents_t size;
	TFunction function;

	IANE_DECL_HYBRID extents_t dimensions() const { return size; }

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID const value_type operator[](const coord_t& aPosition) const
	{
		return function(aPosition);
	}
};

// Return the same stored value at every coordinate.
template<class TValue>
struct constant_pixel {
	TValue value;

	template<class TCoordinate>
	IANE_DECL_HYBRID TValue operator()(const TCoordinate&) const { return value; }
};

// Apply a pointwise callable to a single borrowed operand.
template<class TSource, class TFunction>
struct mapped_pixel {
	TSource source;
	TFunction function;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID auto operator()(const typename TSource::coord_t& aPosition) const
	{
		return function(source[aPosition]);
	}
};

// Apply a pointwise callable to two borrowed operands at the same coordinate.
template<class TLeft, class TRight, class TFunction>
struct zipped_pixel {
	TLeft left;
	TRight right;
	TFunction function;

	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID auto operator()(const typename TLeft::coord_t& aPosition) const
	{
		return function(left[aPosition], right[aPosition]);
	}
};

} // namespace detail

/**
 * @brief Generate values on demand from coordinates.
 * @param aDomain Supplies the dimensions and execution backend. Its storage is unused.
 * @param aFunction Const-invocable coordinate callable stored by value.
 * @return Read-only view with in-bounds indexing and no allocated image storage.
 * @pre CUDA callables must support device execution. Any storage referenced
 * by the callable must remain valid through evaluation and completion.
 */
template<detail::procedural_domain TDomain, class TFunction>
auto generate_view(const TDomain& aDomain, TFunction aFunction)
{
	return detail::procedural_view<TDomain, TFunction>{{}, aDomain.dimensions(), std::move(aFunction)};
}

/**
 * @brief Generate the same value at each coordinate of a domain.
 * @param aDomain Supplies dimensions and backend without lending its storage.
 * @param aValue Value stored in the view and returned by value at each coordinate.
 */
template<detail::procedural_domain TDomain, class TValue>
auto constant_view(const TDomain& aDomain, TValue aValue)
{
	return generate_view(aDomain, detail::constant_pixel<TValue>{std::move(aValue)});
}

/**
 * @brief Lazily apply a const-invocable function to each input pixel.
 * @param aSource Readable input whose view handle is stored by value.
 * @param aFunction Pointwise callable stored by value.
 * @return Read-only view with the source dimensions and backend.
 * @pre Keep borrowed storage valid through evaluation and completion.
 * CUDA callables must support device execution.
 */
template<detail::procedural_domain TSource, class TFunction>
requires concepts::ReadableView<TSource>
auto map_view(TSource aSource, TFunction aFunction)
{
	return generate_view(aSource, detail::mapped_pixel<TSource, TFunction>{aSource, std::move(aFunction)});
}

/**
 * @brief Lazily combine two equal-sized views on the same backend.
 * @param aLeft First readable operand.
 * @param aRight Second readable operand with the same coordinate type and rank.
 * @param aFunction Const-invocable pointwise callable taking the two pixels.
 * @return Read-only view holding both operand handles and the callable by value.
 * @throws std::invalid_argument If extents differ or are negative.
 * @pre Keep both operands' borrowed storage valid through evaluation and
 * completion. CUDA callables must support device execution.
 */
template<detail::procedural_domain TLeft, detail::procedural_domain TRight, class TFunction>
requires (concepts::ReadableView<TLeft> && concepts::ReadableView<TRight> &&
	concepts::MatchingViewRank<TLeft, TRight> && concepts::SameViewBackend<TLeft, TRight>)
auto zip_view(TLeft aLeft, TRight aRight, TFunction aFunction)
{
	const auto left_size = aLeft.dimensions();
	const auto right_size = aRight.dimensions();
	for (int i = 0; i < TLeft::dimension; ++i) {
		if (math::detail::axis(left_size, i) < 0 ||
		    math::detail::axis(left_size, i) != math::detail::axis(right_size, i)) {
			throw std::invalid_argument("Zip operands require matching nonnegative dimensions");
		}
	}
	return generate_view(aLeft, detail::zipped_pixel<TLeft, TRight, TFunction>{
		aLeft, aRight, std::move(aFunction)});
}

} // namespace iane
