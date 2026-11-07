#include <iane/algorithms/convolution.hpp>
#include <iane/view/procedural.hpp>

#include "../common/cli.hpp"
#include "../common/deconvolution.hpp"

#include <algorithm>

namespace {

// Keep the initial estimate positive so multiplicative updates can recover dark pixels.
struct initial_estimate {
	float epsilon;

	IANE_DECL_HYBRID float operator()(std::uint8_t aObserved) const
	{
		return aObserved > epsilon ? static_cast<float>(aObserved) : epsilon;
	}
};

// Form the observed-to-predicted ratio without dividing by zero.
struct observation_ratio {
	float epsilon;

	IANE_DECL_HYBRID float operator()(std::uint8_t aObserved, float aPredicted) const
	{
		return aObserved / (aPredicted > epsilon ? aPredicted : epsilon);
	}
};

// Normalize the backprojection by the response of the adjoint to an all-ones image.
struct edge_correction {
	float epsilon;

	IANE_DECL_HYBRID float operator()(float aBackprojection, float aSensitivity) const
	{
		return aSensitivity > 0.0f
			? aBackprojection / (aSensitivity > epsilon ? aSensitivity : epsilon)
			: 0.0f;
	}
};

// Apply the pointwise multiplicative correction.
struct multiply_estimate {
	IANE_DECL_HYBRID float operator()(float aEstimate, float aCorrection) const
	{
		return aEstimate * aCorrection;
	}
};

template<class TSrc, class TPsf, class TAdjoint, class TDst>
auto prepare_rl(TSrc aSource, TPsf aPsf, TAdjoint aAdjoint, TDst aDestination,
	iane::view_policy<TDst> aPolicy, int aIterations, float aEpsilon)
{
	using image_t = deconvolution::backend_image<float, TDst>;
	const auto size = aDestination.dimensions();
	image_t estimate(size);
	image_t predicted(size);
	image_t correction(size);
	image_t sensitivity(size);
	constexpr iane::convolution_options options{.border = iane::border_mode::zero};
	iane::convolve(iane::constant_view(sensitivity.view(), 1.0f), aAdjoint,
		sensitivity.view(), options, aPolicy).wait();

	return [aSource, aPsf, aAdjoint, aDestination, aPolicy, aIterations, aEpsilon, options,
		estimate = std::move(estimate), predicted = std::move(predicted),
		correction = std::move(correction), sensitivity = std::move(sensitivity)]() mutable {
		iane::transform(aSource, estimate.view(), initial_estimate{aEpsilon}, aPolicy);
		for (int iteration = 0; iteration < aIterations; ++iteration) {
			iane::convolve(estimate.const_view(), aPsf, predicted.view(), options, aPolicy);
			iane::copy(iane::zip_view(aSource, predicted.const_view(), observation_ratio{aEpsilon}),
				predicted.view(), aPolicy);
			iane::convolve(predicted.const_view(), aAdjoint, correction.view(), options, aPolicy);
			const auto factor = iane::zip_view(correction.const_view(), sensitivity.const_view(),
				edge_correction{aEpsilon});
			iane::copy(iane::zip_view(estimate.const_view(), factor, multiply_estimate{}),
				estimate.view(), aPolicy);
		}
		return iane::transform(estimate.const_view(), aDestination, deconvolution::to_gray8{}, aPolicy);
	};
}

} // namespace

int main(int aArgc, char** aArgv)
{
	namespace po = cli::po;
	po::options_description options("Richardson-Lucy options");
	options.add_options()
		("psf", po::value<std::string>()->required(), "Text file containing the point spread function")
		("iterations", po::value<int>()->default_value(20), "Number of iterations per run")
		("epsilon", po::value<float>()->default_value(1e-6f), "Positive floor for divisions");

	return cli::run(aArgc, aArgv, "rl_deconvolution", options,
		[](const std::string& aInputPath, const std::string& aOutputPath,
			const iane::run_options& aExecution, const po::variables_map& aArgs) {
			const int iterations = aArgs["iterations"].as<int>();
			const float epsilon = aArgs["epsilon"].as<float>();
			if (iterations <= 0 || !std::isfinite(epsilon) || epsilon <= 0.0f) {
				throw std::invalid_argument("Iterations and finite epsilon must be positive");
			}
			const auto input = io::read_gray8(aInputPath);
			const auto psf = deconvolution::read_psf(aArgs["psf"].as<std::string>());
			const auto size = psf.dimensions();
			const auto count = static_cast<std::size_t>(size[0]) * size[1];
			iane::host_image<float, 2> flipped(size);
			std::reverse_copy(psf.pointer(), psf.pointer() + count, flipped.pointer());
			iane::host_image<std::uint8_t, 2> output(input.dimensions());
			iane::run_prepared_on(aExecution, std::tuple{input.const_view(), psf.const_view(), flipped.const_view()},
				output.view(), [iterations, epsilon](auto aSource, auto aPsf, auto aAdjoint, auto aDestination,
					auto aExec) {
					return prepare_rl(aSource, aPsf, aAdjoint, aDestination,
						static_cast<iane::view_policy<decltype(aDestination)>>(aExec), iterations, epsilon);
				});
			io::write_gray8(aOutputPath, output);
		});
}
