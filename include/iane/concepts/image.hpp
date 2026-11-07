#pragma once

#include <iane/concepts/view.hpp>

namespace iane::concepts {

/**
 * @ingroup iane_concepts
 * @brief An image-like object exposing a readable constant view.
 * @tparam TImage Candidate image object type.
 *
 * Use this for APIs that take image objects and forward work to a view.
 * Ownership and view lifetime are semantic properties of the concrete type.
 */
template<class TImage>
concept ReadableImage = requires(const TImage& aImage) {
	{ aImage.const_view() } -> ReadableView;
};

/**
 * @ingroup iane_concepts
 * @brief An image-like object also exposing a writable view of matching rank.
 * @tparam TImage Candidate image object type.
 */
template<class TImage>
concept WritableImage = ReadableImage<TImage> && requires(TImage& aImage) {
	{ aImage.view() } -> WritableView;
	requires MatchingViewRank<decltype(aImage.const_view()), decltype(aImage.view())>;
};

} // namespace iane::concepts
