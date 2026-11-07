#pragma once
#include <iane/concepts/view.hpp>

namespace iane::view {

/* ---------------------------------------------------------
   Identity overloads: view(view) → identity
   --------------------------------------------------------- */

template<concepts::AnyView TVector>
inline constexpr TVector view(const TVector &aV)
{
	return aV;
}

template<concepts::AnyView TVector>
inline constexpr TVector const_view(const TVector &aV)
{
	return aV;
}

/* ---------------------------------------------------------
   view(image) → image.view()
   --------------------------------------------------------- */

template<typename TImg>
	requires requires (const TImg &aI) { aI.view(); }
inline auto view(const TImg &aImg)
{
	return aImg.view();
}

template<typename TImg>
	requires requires (const TImg &aI) { aI.const_view(); }
inline auto const_view(const TImg &aImg)
{
	return aImg.const_view();
}

} // namespace iane::view
