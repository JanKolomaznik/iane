#pragma once

#include <iane/algorithms/detail/sequence.hpp>

namespace iane {

/**
 * @brief Configure an inclusive prefix scan over the whole source, with axis 0 fastest.
 * @param aOp Pure associative operation, callable on the selected backend.
 * @param aInit Initial value, applied once before the first element.
 * @return Algorithm for execution_context::prepare with equal source and destination extents.
 *
 * The initial value determines the accumulator and destination element type.
 * Operand order is preserved. Grouping can vary, including for floating-point sums.
 * Supports strided views and readable expressions. Source and destination storage
 * must be disjoint, and destination coordinates must have distinct storage.
 * Empty sources produce no writes. Keep view storage alive through completion.
 */
template<class TOp, class TValue>
auto inclusive_scan(TOp aOp, TValue aInit)
{
	return detail::sequence_algorithm<-1, detail::sequence_kind::inclusive, TOp, TValue>{aOp, aInit};
}

/// Configure an exclusive scan with the same contract as inclusive_scan, omitting the current element.
template<class TOp, class TValue>
auto exclusive_scan(TOp aOp, TValue aInit)
{
	return detail::sequence_algorithm<-1, detail::sequence_kind::exclusive, TOp, TValue>{aOp, aInit};
}

/// Configure independent inclusive scans along axis @p tAxis, each starting at @p aInit.
template<int tAxis, class TOp, class TValue>
requires (tAxis >= 0)
auto inclusive_scan_along(TOp aOp, TValue aInit)
{
	return detail::sequence_algorithm<tAxis, detail::sequence_kind::inclusive, TOp, TValue>{aOp, aInit};
}

/// Configure independent exclusive scans along axis @p tAxis, each starting at @p aInit.
template<int tAxis, class TOp, class TValue>
requires (tAxis >= 0)
auto exclusive_scan_along(TOp aOp, TValue aInit)
{
	return detail::sequence_algorithm<tAxis, detail::sequence_kind::exclusive, TOp, TValue>{aOp, aInit};
}

} // namespace iane
