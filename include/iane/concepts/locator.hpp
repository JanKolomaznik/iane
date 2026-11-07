#pragma once

#include <iane/concepts/core.hpp>

namespace iane::concepts {

/**
 * @ingroup iane_concepts
 * @brief A read-only neighborhood accessor centered at a view coordinate.
 * @tparam TLocator Candidate exposing value_type, coord_t, offset_t and a positive rank.
 *
 * position() returns the center in the source view. Indexing takes a relative
 * offset, so a zero offset selects the center. The locator's border policy
 * defines the value of a sample outside its source view. Use this concept to
 * constrain neighborhood callbacks passed to transform_locator or for_each_locator.
 */
template<class TLocator>
concept Locator = requires(const TLocator& aLocator, const typename TLocator::offset_t& aOffset) {
	typename TLocator::value_type;
	typename TLocator::coord_t;
	requires (TLocator::dimension > 0);
	{ aLocator.position() } -> std::same_as<typename TLocator::coord_t>;
	{ aLocator[aOffset] } -> std::convertible_to<typename TLocator::value_type>;
};

} // namespace iane::concepts
