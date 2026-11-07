#pragma once

#include <iane/concepts/core.hpp>

#include <type_traits>
#include <utility>

namespace iane::concepts {

/**
 * @defgroup iane_view_concepts View concepts
 * @ingroup iane_concepts
 * @brief Separate execution backend, indexing, and storage capabilities.
 *
 * A generated view typically models ReadableView and one backend tag.
 * A view over allocated storage can additionally model MemoryView.
 */

/**
 * @ingroup iane_view_concepts
 * @brief A view tagged for host execution.
 * @tparam TView Candidate exposing the nested is_host_view_tag type.
 */
template<class TView>
concept HostView = requires { typename TView::is_host_view_tag; };

/**
 * @ingroup iane_view_concepts
 * @brief A view tagged for CUDA execution.
 * @tparam TView Candidate exposing the nested is_device_view_tag type.
 *
 * This tag describes where the view is evaluated. It does not indicate
 * whether a CUDA compiler or device is available.
 */
template<class TView>
concept DeviceView = requires { typename TView::is_device_view_tag; };

/**
 * @ingroup iane_view_concepts
 * @brief A view with at least one supported execution tag.
 * @tparam TView Candidate view type.
 */
template<class TView>
concept AnyView = HostView<TView> || DeviceView<TView>;

/**
 * @ingroup iane_view_concepts
 * @brief A tagged view with a positive static rank and a runtime extent.
 * @tparam TView Candidate exposing value_type, coord_t and extents_t.
 *
 * Algorithms use scalar coordinates for rank one and indexable coordinates
 * for higher ranks. The first axis changes fastest in a packed image.
 */
template<class TView>
concept ViewDomain = AnyView<TView> && requires(const TView& aView) {
	typename TView::value_type;
	typename TView::coord_t;
	typename TView::extents_t;
	requires (TView::dimension > 0);
	{ aView.dimensions() } -> std::convertible_to<typename TView::extents_t>;
};

/**
 * @ingroup iane_view_concepts
 * @brief A view whose coordinate lookup yields a value, reference, or readable proxy.
 * @tparam TView Candidate view type.
 */
template<class TView>
concept ReadableView = ViewDomain<TView> &&
	requires(const TView& aView, const typename TView::coord_t& aPosition) {
		{ aView[aPosition] } -> std::convertible_to<typename TView::value_type>;
	};

/**
 * @ingroup iane_view_concepts
 * @brief A view whose indexed result accepts assignment from the given type.
 * @tparam TView Candidate destination view.
 * @tparam TValue Assigned type. Defaults to the view's element type.
 *
 * Assignment can target a reference or a writable proxy. An assignment
 * expression may return void.
 */
template<class TView, class TValue = typename TView::value_type>
concept WritableView = ViewDomain<TView> &&
	requires(const TView& aView, const typename TView::coord_t& aPosition, TValue&& aValue) {
		aView[aPosition] = static_cast<TValue&&>(aValue);
	};

/**
 * @ingroup iane_view_concepts
 * @brief A readable view exposing a typed base pointer and byte strides.
 * @tparam TView Candidate view type.
 *
 * Used for transfers and storage interoperability. Strides may include
 * padding. The pointer is interpreted in the view's tagged memory space.
 */
template<class TView>
concept MemoryView = ReadableView<TView> && requires(const TView& aView) {
	requires (std::same_as<decltype(aView.pointer()), typename TView::value_type*> ||
		std::same_as<decltype(aView.pointer()), const typename TView::value_type*>);
	{ aView.strides() } -> std::convertible_to<typename TView::extents_t>;
};

/**
 * @ingroup iane_view_concepts
 * @brief Views with the same rank and coordinate representation.
 * @tparam TFirst First view type.
 * @tparam TSecond Second view type.
 *
 * Extents are runtime values and may differ even when this concept holds.
 */
template<class TFirst, class TSecond>
concept MatchingViewRank = ViewDomain<TFirst> && ViewDomain<TSecond> &&
	TFirst::dimension == TSecond::dimension &&
	std::same_as<typename TFirst::coord_t, typename TSecond::coord_t>;

/**
 * @ingroup iane_view_concepts
 * @brief Two views share an execution backend.
 * @tparam TFirst First view type.
 * @tparam TSecond Second view type.
 *
 * Useful for operations that consume several inputs, including kernels
 * whose extents may differ from those of the image.
 */
template<class TFirst, class TSecond>
concept SameViewBackend =
	(HostView<TFirst> && HostView<TSecond>) || (DeviceView<TFirst> && DeviceView<TSecond>);

/**
 * @ingroup iane_view_concepts
 * @brief A readable source and writable destination for same-backend computation.
 * @tparam TSrc Source view type.
 * @tparam TDst Destination view type.
 *
 * Element types may differ. The consuming algorithm defines the required
 * relationship between the source value and the destination value.
 */
template<class TSrc, class TDst>
concept TransformViews = MatchingViewRank<TSrc, TDst> && ReadableView<TSrc> && WritableView<TDst> &&
	SameViewBackend<TSrc, TDst>;

/**
 * @ingroup iane_view_concepts
 * @brief Typed host and CUDA memory views suitable for a byte transfer.
 * @tparam TSrc Readable source memory view.
 * @tparam TDst Destination memory view exposing a mutable pointer.
 *
 * Both views have the same trivially copyable element type and rank.
 * Extents and supported stride layouts are checked by the transfer.
 */
template<class TSrc, class TDst>
concept TransferViews = MatchingViewRank<TSrc, TDst> && MemoryView<TSrc> && MemoryView<TDst> &&
	((HostView<TSrc> && DeviceView<TDst>) || (DeviceView<TSrc> && HostView<TDst>)) &&
	std::same_as<typename TSrc::value_type, typename TDst::value_type> &&
	std::is_trivially_copyable_v<typename TSrc::value_type> &&
	requires(const TDst& aDestination) {
		{ aDestination.pointer() } -> std::same_as<typename TDst::value_type*>;
	};

} // namespace iane::concepts
