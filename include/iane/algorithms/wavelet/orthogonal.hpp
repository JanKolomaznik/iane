#pragma once

#include <iane/algorithms/wavelet/separable.hpp>

namespace iane::detail {

// Analyze or synthesize one axis using a periodic orthonormal filter bank.
template<class TFilter, bool tInverse>
struct orthogonal_wavelet {
	template<int tAxis, class TSource>
	struct pass {
		TSource source;
		int half;

		IANE_CUDA_HD_WARNING_DISABLE
		IANE_DECL_HYBRID float operator()(math::vec2i aPosition) const
		{
			const auto index = aPosition[tAxis];
			float result = 0.0f;
			if constexpr (tInverse) {
				// Only taps with this pixel's parity contribute after upsampling.
				for (int tap = index % 2; tap < TFilter::size; tap += 2) {
					const auto coefficient = ((index - tap) / 2) % half;
					aPosition[tAxis] = coefficient < 0 ? coefficient + half : coefficient;
					result += TFilter::low(tap) * source[aPosition];
					aPosition[tAxis] += half;
					result += high(tap) * source[aPosition];
				}
			} else {
				const bool detail_band = index >= half;
				const auto start = 2LL * (detail_band ? index - half : index);
				for (int tap = 0; tap < TFilter::size; ++tap) {
					aPosition[tAxis] = static_cast<int>((start + tap) % (2 * half));
					result += (detail_band ? high(tap) : TFilter::low(tap)) * source[aPosition];
				}
			}
			return result;
		}

		IANE_DECL_HYBRID static float high(int aTap)
		{
			const auto reversed = TFilter::low(TFilter::size - 1 - aTap);
			return aTap % 2 == 0 ? reversed : -reversed;
		}
	};
};

template<class TFilter, bool tInverse>
using orthogonal_wavelet_algorithm = separable_wavelet_algorithm<
	orthogonal_wavelet<TFilter, tInverse>::template pass, !tInverse>;

} // namespace iane::detail
