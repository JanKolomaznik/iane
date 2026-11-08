#pragma once

#include <iane/concepts/view.hpp>

#include <span>
#include <stdexcept>
#include <vector>

namespace iane {

/**
 * @brief Borrow a host table of image views ordered by resolution.
 *
 * Level zero conventionally denotes the finest level. The caller defines
 * extents, layout and relationships between levels. Keep the table and
 * backing images alive while using the view. Pass individual level views
 * to device algorithms and keep backing images alive through completion.
 */
template<concepts::ViewDomain TView>
class pyramid_view {
public:
	/// Borrow a table of level views without allocating or copying descriptors.
	explicit pyramid_view(std::span<const TView> aLevels) : mLevels(aLevels) {}

	/// Return the number of levels in the borrowed table.
	std::size_t level_count() const { return mLevels.size(); }

	/**
	 * @brief Return a copy of the selected level view.
	 * @throws std::out_of_range If @p aIndex is outside the table.
	 */
	TView level(std::size_t aIndex) const
	{
		if (aIndex >= mLevels.size()) {
			throw std::out_of_range("Pyramid level index is out of range");
		}
		return mLevels[aIndex];
	}

private:
	std::span<const TView> mLevels;
};

/**
 * @brief Own independently allocated levels of a single image type.
 * @tparam TImage Owning image exposing extents_t, view_t and const_view_t.
 *
 * The caller supplies the extents and builds the contents using level views.
 * Borrowed descriptors and image storage depend on this owner's lifetime.
 * Keep storage alive through completion of submitted work.
 */
template<class TImage>
class image_pyramid {
public:
	using extents_t = typename TImage::extents_t;
	using view_t = pyramid_view<typename TImage::view_t>;
	using const_view_t = pyramid_view<typename TImage::const_view_t>;

	/// Allocate one image per extent in the supplied order.
	explicit image_pyramid(std::span<const extents_t> aExtents)
	{
		mImages.reserve(aExtents.size());
		mViews.reserve(aExtents.size());
		mConstViews.reserve(aExtents.size());
		for (const auto& extent : aExtents) {
			auto& image = mImages.emplace_back(extent);
			mViews.push_back(image.view());
			mConstViews.push_back(image.const_view());
		}
	}

	image_pyramid(const image_pyramid&) = delete;
	image_pyramid& operator=(const image_pyramid&) = delete;
	image_pyramid(image_pyramid&&) = default;
	image_pyramid& operator=(image_pyramid&&) = default;

	/// Return the number of allocated levels.
	std::size_t level_count() const { return mImages.size(); }

	/// Borrow the table of writable level views from an lvalue owner.
	view_t view() & { return view_t(mViews); }

	/// Borrow the table of read-only level views from an lvalue owner.
	const_view_t const_view() const & { return const_view_t(mConstViews); }
	const_view_t const_view() const && = delete;

private:
	std::vector<TImage> mImages;
	std::vector<typename TImage::view_t> mViews;
	std::vector<typename TImage::const_view_t> mConstViews;
};

} // namespace iane
