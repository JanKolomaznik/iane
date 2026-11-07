#include <iane/algorithms/fft.hpp>
#include <iane/view/procedural.hpp>

#include "../common/cli.hpp"
#include "../common/deconvolution.hpp"

namespace {

using complex_t = iane::fft_complex;
using coord_t = iane::math::vec2i;

// Convert an observation into a spatial complex sample.
struct real_sample {
	IANE_DECL_HYBRID complex_t operator()(std::uint8_t aValue) const
	{
		return {static_cast<float>(aValue), 0.0f};
	}
};

// Apply the conjugate PSF response divided by its regularized power.
struct wiener_factor {
	float noise;

	IANE_DECL_HYBRID complex_t operator()(complex_t aObserved, complex_t aPsf) const
	{
		const double real = aPsf.real;
		const double imag = aPsf.imag;
		const double denominator = real * real + imag * imag + noise;
		return {static_cast<float>((aObserved.real * real + aObserved.imag * imag) / denominator),
			static_cast<float>((aObserved.imag * real - aObserved.real * imag) / denominator)};
	}
};

// Normalize the inverse FFT and encode its real component.
struct encode_sample {
	float scale;

	IANE_DECL_HYBRID std::uint8_t operator()(complex_t aValue) const
	{
		return deconvolution::to_gray8{}(aValue.real * scale);
	}
};

template<class TSrc, class TPsf, class TDst>
auto prepare_wiener(TSrc aSource, TPsf aPsf, TDst aDestination,
	iane::view_policy<TDst> aPolicy, float aNoise)
{
	using image_t = deconvolution::backend_image<complex_t, TDst>;
	using policy_t = iane::view_policy<TDst>;
	using context_t = iane::execution_context<policy_t>;
	using plan_t = decltype(std::declval<context_t&>().prepare(iane::fft_forward,
		std::declval<typename image_t::const_view_t>(), std::declval<typename image_t::view_t>()));

	// Keep the plans, context and image buffers alive in dependency order.
	struct work_images {
		image_t spatial;
		image_t spectrum;
		image_t response;
		context_t context;
		plan_t forward;
		plan_t inverse;

		work_images(coord_t aSize, policy_t aPolicy)
			: spatial(aSize), spectrum(aSize), response(aSize), context(aPolicy),
			  forward(context.prepare(iane::fft_forward, spatial.const_view(), spectrum.view())),
			  inverse(context.prepare(iane::fft_inverse, spectrum.const_view(), spatial.view())) {}
	};

	const auto size = aDestination.dimensions();
	auto work = std::make_unique<work_images>(size, aPolicy);
	work->context.run(work->forward, aPsf, work->response.view()).wait();
	const auto scale = static_cast<float>(1.0 / (static_cast<double>(size[0]) * size[1]));

	return [aSource, aDestination, aPolicy, aNoise, scale, work = std::move(work)] {
		auto& [spatial, spectrum, response, context, forward, inverse] = *work;
		iane::transform(aSource, spatial.view(), real_sample{}, aPolicy);
		context.run(forward, spatial.const_view(), spectrum.view());
		iane::copy(iane::zip_view(spectrum.const_view(), response.const_view(), wiener_factor{aNoise}),
			spectrum.view(), aPolicy);
		context.run(inverse, spectrum.const_view(), spatial.view());
		return iane::transform(spatial.const_view(), aDestination, encode_sample{scale}, aPolicy);
	};
}

} // namespace

int main(int aArgc, char** aArgv)
{
	namespace po = cli::po;
	po::options_description options("Wiener options");
	options.add_options()
		("psf", po::value<std::string>()->required(), "Text file containing the point spread function")
		("noise", po::value<float>()->default_value(0.002f), "Positive Wiener regularization constant");

	return cli::run(aArgc, aArgv, "wiener", options,
		[](const std::string& aInputPath, const std::string& aOutputPath,
			const iane::run_options& aExecution, const po::variables_map& aArgs) {
			const float noise = aArgs["noise"].as<float>();
			if (!std::isfinite(noise) || noise <= 0.0f) {
				throw std::invalid_argument("Noise must be finite and positive");
			}
			const auto input = io::read_gray8(aInputPath);
			const auto psf = deconvolution::read_psf(aArgs["psf"].as<std::string>());
			const auto size = input.dimensions();
			const auto extent = psf.dimensions();
			if (extent[0] > size[0] || extent[1] > size[1]) {
				throw std::invalid_argument("PSF must fit inside the input image");
			}
			iane::host_image<complex_t, 2> padded(size);
			iane::for_each(padded.view(), [](complex_t& aValue) { aValue = {}; });
			auto padded_view = padded.view();
			const auto kernel = psf.const_view();
			for (int y = 0; y < extent[1]; ++y) {
				for (int x = 0; x < extent[0]; ++x) {
					const int u = x - extent[0] / 2;
					const int v = y - extent[1] / 2;
					padded_view[coord_t{u < 0 ? u + size[0] : u, v < 0 ? v + size[1] : v}] =
						{kernel[coord_t{x, y}], 0.0f};
				}
			}
			iane::host_image<std::uint8_t, 2> output(size);
			iane::run_prepared_on(aExecution, std::tuple{input.const_view(), padded.const_view()}, output.view(),
				[noise](auto aSource, auto aPsf, auto aDestination, auto aExec) {
					return prepare_wiener(aSource, aPsf, aDestination,
						static_cast<iane::view_policy<decltype(aDestination)>>(aExec), noise);
				});
			io::write_gray8(aOutputPath, output);
		});
}
