#include <iane/algorithms/wavelet/db2.hpp>
#include <iane/algorithms/wavelet/haar.hpp>
#include <iane/algorithms/wavelet/multilevel.hpp>
#include <iane/algorithms/wavelet/sym4.hpp>

#if defined(IANE_EXAMPLE_FFT)
#include <iane/algorithms/fft/rfft.hpp>
#include <iane/algorithms/fft/spectrum.hpp>

#include <filesystem>
#include <numbers>
#endif

#include "../common/cli.hpp"
#include "../common/io.hpp"

#include <cmath>
#include <limits>

namespace {

using coord_t = iane::math::vec2i;
using gray_image = iane::host_image<std::uint8_t, 2>;

template<class TAlgorithm, class TSrc, class TDst>
void run_transform(const iane::run_options& aExecution, TAlgorithm aAlgorithm, TSrc aSource, TDst aDestination)
{
	iane::run_prepared_on(aExecution, std::tuple{aSource}, aDestination,
		[aAlgorithm](auto aInput, auto aOutput, auto aExec) {
			using policy_t = iane::view_policy<decltype(aOutput)>;
			using context_t = iane::execution_context<policy_t>;
			using plan_t = decltype(std::declval<context_t&>().prepare(aAlgorithm, aInput, aOutput));

			// Destroy the prepared plan before the context that owns its scratch.
			struct prepared_transform {
				context_t context;
				plan_t plan;

				prepared_transform(policy_t aPolicy, TAlgorithm aTransform,
					decltype(aInput) aSourceView, decltype(aOutput) aOutputView)
					: context(aPolicy), plan(context.prepare(aTransform, aSourceView, aOutputView)) {}
			};

			auto work = std::make_unique<prepared_transform>(
				static_cast<policy_t>(aExec), aAlgorithm, aInput, aOutput);
			return [aInput, aOutput, work = std::move(work)] {
				return work->context.run(work->plan, aInput, aOutput);
			};
		});
}

iane::host_image<float, 2> make_float_input(const gray_image& aInput, int aBlock)
{
	const auto size = aInput.dimensions();
	coord_t padded{};
	for (int axis = 0; axis < 2; ++axis) {
		const auto extent = ((static_cast<std::int64_t>(size[axis]) + aBlock - 1) / aBlock) * aBlock;
		if (extent > std::numeric_limits<int>::max()) {
			throw std::invalid_argument("Requested wavelet levels exceed supported image extents");
		}
		padded[axis] = static_cast<int>(extent);
	}

	iane::host_image<float, 2> result(padded);
	const auto input = aInput.const_view();
	iane::copy(iane::generate_view(result.view(), [input, size](coord_t aPosition) {
		return static_cast<float>(input[coord_t{
			std::min(aPosition[0], size[0] - 1), std::min(aPosition[1], size[1] - 1)}]);
	}), result.view());
	return result;
}

std::uint8_t encode_byte(float aValue)
{
	return static_cast<std::uint8_t>(std::lround(std::clamp(aValue, 0.0f, 255.0f)));
}

template<class TForward>
void save_wavelet(const gray_image& aInput, const std::string& aOutput,
	const iane::run_options& aExecution, TForward aForward, int aLevels)
{
	if (aLevels <= 0 || aLevels >= std::numeric_limits<int>::digits) {
		throw std::invalid_argument("Wavelet levels must be in [1, 30]");
	}
	const int block = 1 << aLevels;
	const auto spatial = make_float_input(aInput, block);
	iane::host_image<float, 2> coefficients(spatial.dimensions());
	run_transform(aExecution, iane::multilevel_wavelet_forward(aForward, aLevels),
		spatial.const_view(), coefficients.view());

	gray_image output(coefficients.dimensions());
	const auto source = iane::make_wavelet_pyramid_view(coefficients.const_view(), aLevels);
	const auto destination = iane::make_wavelet_pyramid_view(output.view(), aLevels);
	iane::transform(source.approximation(), destination.approximation(),
		[block](float aValue) { return encode_byte(aValue / block); });
	for (const auto band : {iane::wavelet_band::hl, iane::wavelet_band::lh, iane::wavelet_band::hh}) {
		const auto input_band = source.details(band);
		const auto output_band = destination.details(band);
		for (std::size_t i = 0; i < input_band.level_count(); ++i) {
			float maximum = 0.0f;
			iane::for_each(input_band.level(i), [&](float aValue) {
				maximum = std::max(maximum, std::abs(aValue));
			});
			// Suppress floating-point residue on flat bands at the current scale.
			const float noise_floor = std::ldexp(16.0f * 255.0f * std::numeric_limits<float>::epsilon(),
				static_cast<int>(i));
			iane::transform(input_band.level(i), output_band.level(i), [maximum, noise_floor](float aValue) {
				return encode_byte(maximum > noise_floor ? 127.5f * (1.0f + aValue / maximum) : 128.0f);
			});
		}
	}
	io::write_gray8(aOutput, output);
}

#if defined(IANE_EXAMPLE_FFT)
void save_fft(const gray_image& aInput, const std::string& aOutput,
	const std::string& aPhaseOutput, const iane::run_options& aExecution)
{
	namespace fs = std::filesystem;
	const fs::path amplitude_path(aOutput);
	const auto phase_path = aPhaseOutput.empty() ?
		amplitude_path.parent_path() / (amplitude_path.stem().string() + "_phase" +
			amplitude_path.extension().string()) : fs::path(aPhaseOutput);
	if (fs::weakly_canonical(fs::absolute(amplitude_path)) ==
		fs::weakly_canonical(fs::absolute(phase_path))) {
		throw std::invalid_argument("Amplitude and phase output paths must differ");
	}

	const auto spatial = make_float_input(aInput, 1);
	const auto size = spatial.dimensions();
	iane::host_image<iane::fft_complex, 2> packed(iane::rfft_extent(size));
	run_transform(aExecution, iane::rfft, spatial.const_view(), packed.view());

	const auto centered = iane::fftshift_view(iane::hermitian_spectrum_view(packed.const_view(), size));
	const auto amplitude = iane::log_magnitude_view(centered);
	float maximum = 0.0f;
	iane::for_each(amplitude, [&](float aValue) { maximum = std::max(maximum, aValue); });
	gray_image output(size);
	iane::transform(amplitude, output.view(), [maximum](float aValue) {
		return encode_byte(maximum > 0.0f ? 255.0f * aValue / maximum : 0.0f);
	});
	io::write_gray8(aOutput, output);

	iane::transform(centered, output.view(), [](iane::fft_complex aValue) {
		const float angle = aValue.real == 0.0f && aValue.imag == 0.0f ?
			0.0f : std::atan2(aValue.imag, aValue.real);
		return encode_byte(127.5f * (1.0f + angle / std::numbers::pi_v<float>));
	});
	io::write_gray8(phase_path.string(), output);
}
#endif

} // namespace

int main(int aArgc, char** aArgv)
{
	namespace po = cli::po;
	po::options_description options("Spectral transform options");
	options.add_options()
		("transform", po::value<std::string>()->default_value("haar"), "fft, haar, db2 or sym4")
		("levels", po::value<int>()->default_value(2), "Number of wavelet decomposition levels")
		("phase-output", po::value<std::string>()->default_value(""), "FFT phase output path (default: OUTPUT with _phase suffix)");

	return cli::run(aArgc, aArgv, "spectral_transform", options,
		[](const std::string& aInput, const std::string& aOutput,
			const iane::run_options& aExecution, const po::variables_map& aArgs) {
			const auto& transform = aArgs["transform"].as<std::string>();
			if (transform != "fft" && transform != "haar" && transform != "db2" && transform != "sym4") {
				throw std::invalid_argument("Transform must be fft, haar, db2 or sym4");
			}
#if !defined(IANE_EXAMPLE_FFT)
			if (transform == "fft") {
				throw std::invalid_argument("FFT mode requires a build with IANE_ENABLE_FFT=ON");
			}
#endif
			const auto input = io::read_gray8(aInput);
			const int levels = aArgs["levels"].as<int>();
			if (transform == "haar") {
				save_wavelet(input, aOutput, aExecution, iane::wavelet_forward, levels);
			} else if (transform == "db2") {
				save_wavelet(input, aOutput, aExecution, iane::db2_forward, levels);
			} else if (transform == "sym4") {
				save_wavelet(input, aOutput, aExecution, iane::sym4_forward, levels);
#if defined(IANE_EXAMPLE_FFT)
			} else {
				save_fft(input, aOutput, aArgs["phase-output"].as<std::string>(), aExecution);
#endif
			}
		});
}
