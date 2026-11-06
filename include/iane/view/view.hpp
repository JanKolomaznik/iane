#pragma once
#include <iane/concepts/view.hpp>

namespace iane::view {

/* ---------------------------------------------------------
   Identity overloads: view(view) → identity
   --------------------------------------------------------- */

template<concepts::AnyView V>
inline constexpr V view(const V &v)
{
	return v;
}

template<concepts::AnyView V>
inline constexpr V const_view(const V &v)
{
	return v;
}

/* ---------------------------------------------------------
   view(image) → image.view()
   --------------------------------------------------------- */

template<typename Img>
	requires requires (const Img &i) { i.view(); }
inline auto view(const Img &img)
{
	return img.view();
}

template<typename Img>
	requires requires (const Img &i) { i.const_view(); }
inline auto const_view(const Img &img)
{
	return img.const_view();
}

} // namespace iane::view
