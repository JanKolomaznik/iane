#pragma once

#include <iane/concepts/locator.hpp>
#include <iane/concepts/view.hpp>
#include <iane/config.hpp>
#include <iane/math/vector_ops.hpp>

#include <utility>

namespace iane {

/**
 * @brief Borrow an interpolated view for sampling around a continuous center.
 * @tparam TView Source interpolated view.
 *
 * The center is relative to the view's outer corner. Sampling uses relative
 * offsets and inherits the source's interpolation and boundaries. Keep the
 * source storage alive throughout sampling and queued work.
 */
template<concepts::InterpolatedView TView>
class sampling_image_locator {
public:
	using coord_t = typename TView::sample_coord_t;
	using offset_t = coord_t;
	using value_type = std::remove_cvref_t<decltype(
		std::declval<const TView&>().sample(std::declval<const coord_t&>()))>;
	static constexpr int dimension = TView::dimension;

	/// Borrow a source at a continuous center in source-view coordinates.
	IANE_DECL_HYBRID sampling_image_locator(TView aSource, coord_t aCenter)
		: mSource(aSource), mCenter(aCenter) {}

	/// Return the continuous center in source-view coordinates.
	IANE_DECL_HYBRID coord_t position() const { return mCenter; }

	/// Sample the source at position() plus the given offset.
	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID value_type sample(const offset_t& aOffset) const
	{
		return mSource.sample(mCenter + aOffset);
	}

private:
	TView mSource;
	coord_t mCenter;
};

} // namespace iane
