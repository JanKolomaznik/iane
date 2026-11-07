#pragma once

#include <iane/concepts/view.hpp>
#include <iane/concepts/locator.hpp>
#include <iane/config.hpp>
#include <iane/math/index.hpp>

#include <concepts>
#include <cstdint>

namespace iane {

/// Sampling rule for coordinates outside a view.
enum class border_mode { clamp, zero };

/**
 * @brief Read values at offsets relative to a position in a nonempty view.
 * @tparam TView A readable image view with scalar or vector coordinates.
 *
 * The locator borrows the view's storage and treats the view's extent as
 * the boundary, including when the view represents a crop.
 */
template<concepts::ReadableView TView>
class image_locator {
public:
	using value_type = typename TView::value_type;
	using coord_t = typename TView::coord_t;
	using offset_t = coord_t;
	static constexpr int dimension = TView::dimension;

	/// Create a locator centered on a valid position in the source view.
	IANE_DECL_HYBRID image_locator(TView aSource, coord_t aPosition, border_mode aBorder)
		: mSource(aSource), mPosition(aPosition), mBorder(aBorder) {}

	/// Return the center in source-view coordinates.
	IANE_DECL_HYBRID coord_t position() const { return mPosition; }

	/// Read a sample at a relative offset, applying the configured border rule.
	IANE_CUDA_HD_WARNING_DISABLE
	IANE_DECL_HYBRID value_type operator[](const offset_t& aOffset) const
	{
		const auto size = mSource.dimensions();
		coord_t sample{};
		for (int axis = 0; axis < dimension; ++axis) {
			const auto coordinate = static_cast<std::int64_t>(math::detail::axis(mPosition, axis)) + math::detail::axis(aOffset, axis);
			if (coordinate >= 0 && coordinate < math::detail::axis(size, axis)) {
				math::detail::axis(sample, axis) = static_cast<int>(coordinate);
			} else {
				if (mBorder == border_mode::zero || math::detail::axis(size, axis) <= 0) {
					return {};
				}
				math::detail::axis(sample, axis) = coordinate < 0 ? 0 : math::detail::axis(size, axis) - 1;
			}
		}
		return mSource[sample];
	}

private:
	TView mSource;
	coord_t mPosition;
	border_mode mBorder;
};

} // namespace iane
