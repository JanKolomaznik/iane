#pragma once

#include <iane/algorithms/detail/sequence.hpp>

namespace iane {

/**
 * @brief Configure a reduction of the entire source into a 1D view of extent one.
 * @param aOp Pure associative and commutative operation, callable on the selected backend.
 * @param aInit Initial value, applied once and returned for an empty source.
 * @return Algorithm for execution_context::prepare on a single backend.
 *
 * The initial value determines the accumulator and destination element type.
 * Grouping can vary, including for floating-point sums. Supports strided views
 * and readable expressions. Source and destination storage must be disjoint.
 * Keep view storage alive through completion.
 */
template<class TOp, class TValue>
auto reduce(TOp aOp, TValue aInit)
{
	return detail::sequence_algorithm<-1, detail::sequence_kind::reduction, TOp, TValue>{aOp, aInit};
}

/**
 * @brief Configure independent reductions along axis @p tAxis.
 *
 * The output shape removes the reduced axis. Rank-one inputs require a
 * rank-one output of extent one. Empty lines produce @p aInit. The reduction
 * contract applies to each line, and output coordinates must have distinct storage.
 */
template<int tAxis, class TOp, class TValue>
requires (tAxis >= 0)
auto reduce_along(TOp aOp, TValue aInit)
{
	return detail::sequence_algorithm<tAxis, detail::sequence_kind::reduction, TOp, TValue>{aOp, aInit};
}

} // namespace iane
