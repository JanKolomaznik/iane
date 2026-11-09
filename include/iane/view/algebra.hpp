#pragma once

#include <iane/view/procedural.hpp>

#include <cmath>

namespace iane {
namespace detail {

template<class TView>
concept algebra_view = procedural_domain<TView> && concepts::ReadableView<TView>;

template<class TLeft, class TRight>
concept algebra_operands =
	(algebra_view<TLeft> && algebra_view<TRight> &&
		concepts::MatchingViewRank<TLeft, TRight> && concepts::SameViewBackend<TLeft, TRight>) ||
	(algebra_view<TLeft> && std::is_arithmetic_v<TRight>) ||
	(std::is_arithmetic_v<TLeft> && algebra_view<TRight>);

template<class TOperand, algebra_view TDomain>
auto broadcast_operand(TOperand aOperand, const TDomain& aDomain)
{
	if constexpr (algebra_view<TOperand>) {
		return aOperand;
	} else {
		return constant_view(aDomain, aOperand);
	}
}

template<class TLeft, class TRight, class TFunction>
auto arithmetic_view(TLeft aLeft, TRight aRight, TFunction aFunction)
{
	if constexpr (algebra_view<TLeft>) {
		return zip_view(aLeft, broadcast_operand(aRight, aLeft), aFunction);
	} else {
		return zip_view(constant_view(aRight, aLeft), aRight, aFunction);
	}
}

// Apply unary negation with the pixel type's C++ promotion rules.
struct negate_view_pixel {
	template<class TValue>
	IANE_DECL_HYBRID auto operator()(const TValue& aValue) const -> decltype(-aValue)
	{
		return -aValue;
	}
};

} // namespace detail

/// Lazily negate each pixel. See map_view for storage and lifetime requirements.
template<detail::algebra_view TView>
auto operator-(TView aView)
{
	return map_view(aView, detail::negate_view_pixel{});
}

// Define each operator together with its device-callable pixel operation.
#define IANE_DEFINE_VIEW_OPERATOR(tOperator, TOperation) \
	namespace detail { \
	/* Apply the corresponding C++ operator to two pixels. */ \
	struct TOperation { \
		template<class TLeft, class TRight> \
		IANE_DECL_HYBRID auto operator()(const TLeft& aLeft, const TRight& aRight) const \
			-> decltype(aLeft tOperator aRight) \
		{ \
			return aLeft tOperator aRight; \
		} \
	}; \
	} \
	template<class TLeft, class TRight> \
	requires detail::algebra_operands<TLeft, TRight> \
	auto operator tOperator(TLeft aLeft, TRight aRight) \
	{ \
		return detail::arithmetic_view(aLeft, aRight, detail::TOperation{}); \
	}

/**
 * @name Pointwise view arithmetic
 * Combine two readable views or broadcast an arithmetic scalar on either side.
 * Results are evaluated lazily using the pixel's C++ arithmetic operators.
 * See zip_view for layout, lifetime, backend and shape requirements.
 * @{
 */
IANE_DEFINE_VIEW_OPERATOR(+, add_view_pixels)
IANE_DEFINE_VIEW_OPERATOR(-, subtract_view_pixels)
IANE_DEFINE_VIEW_OPERATOR(*, multiply_view_pixels)
IANE_DEFINE_VIEW_OPERATOR(/, divide_view_pixels)
/// @}

#undef IANE_DEFINE_VIEW_OPERATOR

// Define each named unary function with a device-callable scalar operation.
#define IANE_DEFINE_VIEW_MATH(tFunction) \
	namespace detail { \
	/* Apply the scalar standard-library math function to one pixel. */ \
	struct tFunction##_view_pixel { \
		template<class TValue> \
		IANE_DECL_HYBRID auto operator()(const TValue& aValue) const -> decltype(std::tFunction(aValue)) \
		{ \
			return std::tFunction(aValue); \
		} \
	}; \
	} \
	template<detail::algebra_view TView> \
	requires std::invocable<detail::tFunction##_view_pixel, typename TView::value_type> \
	auto tFunction(TView aView) \
	{ \
		return map_view(aView, detail::tFunction##_view_pixel{}); \
	}

/**
 * @name Pointwise view math
 * Lazily apply the corresponding standard math function, including its
 * scalar type promotions and domain rules. See map_view for lifetimes.
 * @{
 */
IANE_DEFINE_VIEW_MATH(log)
IANE_DEFINE_VIEW_MATH(log1p)
IANE_DEFINE_VIEW_MATH(log2)
IANE_DEFINE_VIEW_MATH(log10)
IANE_DEFINE_VIEW_MATH(exp)
IANE_DEFINE_VIEW_MATH(exp2)
IANE_DEFINE_VIEW_MATH(sqrt)
IANE_DEFINE_VIEW_MATH(sin)
IANE_DEFINE_VIEW_MATH(cos)
/// @}

#undef IANE_DEFINE_VIEW_MATH

namespace detail {

// Raise one pixel to the corresponding exponent.
struct pow_view_pixels {
	template<class TBase, class TExponent>
	IANE_DECL_HYBRID auto operator()(const TBase& aBase, const TExponent& aExponent) const
		-> decltype(std::pow(aBase, aExponent))
	{
		return std::pow(aBase, aExponent);
	}
};

} // namespace detail

/// Lazily apply std::pow to corresponding pixels or broadcast an arithmetic scalar.
template<class TBase, class TExponent>
requires detail::algebra_operands<TBase, TExponent>
auto pow(TBase aBase, TExponent aExponent)
{
	return detail::arithmetic_view(aBase, aExponent, detail::pow_view_pixels{});
}

namespace detail {
// Make operators discoverable by ADL even when both operands are procedural.
using iane::operator+;
using iane::operator-;
using iane::operator*;
using iane::operator/;
} // namespace detail

} // namespace iane
