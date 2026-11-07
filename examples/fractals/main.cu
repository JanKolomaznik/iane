#include <iane/algorithms/for_each.hpp>

#include "../common/cli.hpp"
#include "../common/io.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace po = cli::po;

namespace {

enum class fractal_type { mandelbrot, julia, burning_ship };

// Parameters for mapping one output pixel to the complex plane and coloring it.
struct render_pixel {
	fractal_type type;
	double left;
	double top;
	double dx;
	double dy;
	double c_real;
	double c_imag;
	double escape_squared;
	int iterations;
	int period;
	int color_count;
	io::rgb8_pixel inside;
	io::rgb8_pixel colors[32];

	IANE_DECL_HYBRID std::uint8_t blend(int aLeft, int aRight, int aWeight) const
	{
		const auto sum = static_cast<std::int64_t>(aLeft) * (period - aWeight)
			+ static_cast<std::int64_t>(aRight) * aWeight;
		return static_cast<std::uint8_t>((sum + period / 2) / period);
	}

	IANE_DECL_HYBRID void operator()(io::rgb8_pixel& aPixel,
		const iane::math::dim_traits<2>::coord_t& aPosition) const
	{
		const double real = ::fma(aPosition[0] + 0.5, dx, left);
		const double imag = ::fma(aPosition[1] + 0.5, dy, top);
		const bool julia = type == fractal_type::julia;
		const double cr = julia ? c_real : real;
		const double ci = julia ? c_imag : imag;
		double x = julia ? real : 0;
		double y = julia ? imag : 0;
		int iteration = 0;
		for (; iteration <= iterations; ++iteration) {
			const double xx = x * x;
			const double yy = y * y;
			if (xx + yy > escape_squared) {
				break;
			}
			if (iteration == iterations) {
				aPixel = inside;
				return;
			}
			const double xy = x * y;
			y = ::fma(2.0, type == fractal_type::burning_ship ? ::fabs(xy) : xy, ci);
			x = (xx - yy) + cr;
		}

		const auto cycle = static_cast<std::int64_t>(iteration % period) * color_count;
		const auto index = static_cast<int>(cycle / period);
		const auto weight = static_cast<int>(cycle % period);
		const auto first = colors[index];
		const auto next = colors[(index + 1) % color_count];
		aPixel = {blend(first.r, next.r, weight), blend(first.g, next.g, weight),
			blend(first.b, next.b, weight)};
	}
};

io::rgb8_pixel parse_color(std::string_view aText)
{
	if (aText.starts_with('#')) {
		aText.remove_prefix(1);
	}
	unsigned int rgb{};
	if (aText.size() != 6) {
		throw std::invalid_argument("Expected a six-digit RGB color");
	}
	const auto end = aText.data() + aText.size();
	const auto result = std::from_chars(aText.data(), end, rgb, 16);
	if (result.ec != std::errc{} || result.ptr != end) {
		throw std::invalid_argument("Expected a six-digit RGB color");
	}
	return {static_cast<std::uint8_t>(rgb >> 16), static_cast<std::uint8_t>(rgb >> 8),
		static_cast<std::uint8_t>(rgb)};
}

std::vector<std::string> palette(const po::variables_map& aArgs)
{
	if (aArgs.count("colors")) {
		return aArgs["colors"].as<std::vector<std::string>>();
	}
	const auto& name = aArgs["palette"].as<std::string>();
	if (name == "ocean") {
		return {"050b28", "103c79", "36b4bd", "f5edaf"};
	}
	if (name == "ember") {
		return {"260b25", "9c1538", "ff6a24", "fff0a0"};
	}
	if (name == "gray") {
		return {"202020", "ffffff"};
	}
	throw std::invalid_argument("Palette must be ocean, ember or gray");
}

fractal_type parse_type(const std::string& aName)
{
	if (aName == "mandelbrot") {
		return fractal_type::mandelbrot;
	}
	if (aName == "julia") {
		return fractal_type::julia;
	}
	if (aName == "burning-ship") {
		return fractal_type::burning_ship;
	}
	throw std::invalid_argument("Type must be mandelbrot, julia or burning-ship");
}

std::array<double, 2> complex_value(const po::variables_map& aArgs, const char* aKey)
{
	const auto& values = aArgs[aKey].as<std::vector<double>>();
	if (values.size() != 2 ||
	    !(std::hypot(values[0], values[1]) <= std::numeric_limits<double>::max() / 4)) {
		throw std::invalid_argument(std::string(aKey) + " requires two finite complex components");
	}
	return {values[0], values[1]};
}

std::string frame_path(const std::string& aOutput, int aFrame, int aFrames)
{
	if (aFrames == 1) {
		return aOutput;
	}
	const std::filesystem::path path(aOutput);
	const auto index = std::to_string(aFrame);
	const auto digits = std::max(std::size_t{4}, std::to_string(aFrames - 1).size());
	return (path.parent_path() / (path.stem().string() + "_" +
		std::string(digits - index.size(), '0') + index + path.extension().string())).string();
}

void render(const std::string& aOutput, const iane::run_options& aExecution, const po::variables_map& aArgs)
{
	const int width = aArgs["width"].as<int>();
	const int height = aArgs["height"].as<int>();
	const int frames = aArgs["frames"].as<int>();
	render_pixel pixel{};
	pixel.type = parse_type(aArgs["type"].as<std::string>());
	pixel.iterations = aArgs["iterations"].as<int>();
	pixel.period = aArgs["color-period"].as<int>();
	if (width <= 0 || height <= 0 || frames <= 0 || pixel.iterations <= 0 || pixel.period <= 0) {
		throw std::invalid_argument("Resolution, frames, iterations and color period must be positive");
	}
	if ((frames > 1 && (pixel.type != fractal_type::julia || !aArgs.count("c-end"))) ||
	    (frames == 1 && aArgs.count("c-end"))) {
		throw std::invalid_argument("A frame sequence requires Julia, --frames > 1 and --c-end");
	}
	if (pixel.type != fractal_type::julia && !aArgs["c"].defaulted()) {
		throw std::invalid_argument("--c applies only to Julia");
	}
	const auto start = complex_value(aArgs, "c");
	const auto end = frames > 1 ? complex_value(aArgs, "c-end") : start;
	const auto& bounds = aArgs["bounds"].as<std::vector<double>>();
	if (bounds.size() != 4 || !std::isfinite(bounds[2] - bounds[0]) ||
	    !std::isfinite(bounds[3] - bounds[1]) || bounds[0] >= bounds[2] || bounds[1] >= bounds[3]) {
		throw std::invalid_argument("Bounds must be finite XMIN YMIN XMAX YMAX with positive spans");
	}
	pixel.left = bounds[0];
	pixel.top = bounds[3];
	pixel.dx = (bounds[2] - bounds[0]) / width;
	pixel.dy = (bounds[1] - bounds[3]) / height;
	pixel.inside = parse_color(aArgs["inside"].as<std::string>());
	const auto colors = palette(aArgs);
	if (colors.size() < 2 || colors.size() > std::size(pixel.colors)) {
		throw std::invalid_argument("A palette needs 2 to 32 colors");
	}
	pixel.color_count = static_cast<int>(colors.size());
	for (int i = 0; i < pixel.color_count; ++i) {
		pixel.colors[i] = parse_color(colors[i]);
	}

	iane::host_image<io::rgb8_pixel, 2> output(width, height);
	for (int frame = 0; frame < frames; ++frame) {
		const double fraction = frames == 1 ? 0.0 : static_cast<double>(frame) / (frames - 1);
		pixel.c_real = std::lerp(start[0], end[0], fraction);
		pixel.c_imag = std::lerp(start[1], end[1], fraction);
		const auto radius = pixel.type == fractal_type::julia
			? std::max(2.0, 0.5 + std::sqrt(0.25 + std::hypot(pixel.c_real, pixel.c_imag))) : 2.0;
		pixel.escape_squared = radius * radius;
		const auto path = frame_path(aOutput, frame, frames);
		std::cout << "frame=" << frame << " output=" << path << '\n';
		iane::run_on(aExecution, output.view(), [&](auto aDst, auto& aExec) {
			return iane::for_each_position(aDst, pixel, aExec);
		});
		io::write_rgb8(path, output);
	}
}

} // namespace

int main(int aArgc, char** aArgv)
{
	po::options_description options("Fractal options");
	options.add_options()
		("type", po::value<std::string>()->default_value("mandelbrot"), "mandelbrot, julia or burning-ship")
		("width", po::value<int>()->default_value(1200), "Output width")
		("height", po::value<int>()->default_value(800), "Output height")
		("bounds", po::value<std::vector<double>>()->multitoken()
			->default_value({-2.5, -1.25, 1.25, 1.25}, "-2.5 -1.25 1.25 1.25"),
			"Viewport as XMIN YMIN XMAX YMAX")
		("iterations", po::value<int>()->default_value(256), "Maximum iterations per pixel")
		("palette", po::value<std::string>()->default_value("ocean"), "ocean, ember or gray")
		("colors", po::value<std::vector<std::string>>()->multitoken(), "Custom palette of 2 to 32 RRGGBB colors")
		("color-period", po::value<int>()->default_value(64), "Escape iterations per palette cycle")
		("inside", po::value<std::string>()->default_value("000000"), "Color for points that do not escape")
		("c", po::value<std::vector<double>>()->multitoken()->default_value({-0.8, 0.156}, "-0.8 0.156"),
			"Julia parameter as REAL IMAG")
		("c-end", po::value<std::vector<double>>()->multitoken(), "Final Julia parameter as REAL IMAG")
		("frames", po::value<int>()->default_value(1), "Number of frames");
	return cli::run<false>(aArgc, aArgv, "fractals", options, render);
}
