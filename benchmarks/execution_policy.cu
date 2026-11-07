#include <iane/execution/run_on.hpp>
#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

// Mix each input independently using defined unsigned arithmetic on both backends.
struct mix_pixel {
	IANE_DECL_HYBRID std::uint32_t operator()(std::uint32_t aValue) const
	{
		for (unsigned round = 0; round < 48; ++round) {
			aValue ^= aValue >> 16;
			aValue *= 0x7feb352du;
			aValue ^= aValue >> 15;
			aValue = aValue * 0x846ca68bu + round;
		}
		return aValue;
	}
};

void run_benchmark()
{
	constexpr int width = 4096;
	constexpr int height = 2048;
	constexpr auto count = static_cast<std::size_t>(width) * height;
	using image_t = iane::host_image<std::uint32_t, 2>;
	const image_t::extents_t size{width, height};
	image_t input(size);
	image_t reference(size);
	image_t output(size);
	for (std::size_t i = 0; i < count; ++i) {
		input.pointer()[i] = static_cast<std::uint32_t>(i);
	}

	const auto run = [&](const char* aLabel, iane::backend aBackend, iane::cpu_policy aPolicy,
		image_t& aDestination) {
		std::cout << aLabel << '\n';
		iane::run_on(iane::run_options{.backend = aBackend, .warmup = 1, .runs = 3},
			input.const_view(), aDestination.view(), [aPolicy](auto aSource, auto aDst, auto aExecution) {
				if constexpr (iane::concepts::HostView<decltype(aDst)>) {
					return iane::transform(aSource, aDst, mix_pixel{}, aPolicy);
				} else {
#if defined(__CUDACC__)
					return iane::transform(aSource, aDst, mix_pixel{},
						iane::cuda_policy{.stream = aExecution.stream});
#endif
				}
			});
		if (!std::equal(reference.pointer(), reference.pointer() + count, aDestination.pointer())) {
			throw std::runtime_error("Benchmark result differs from the serial result");
		}
	};
	run("serial CPU", iane::backend::cpu, {}, reference);
	run("CPU: 2 workers, grain 16384", iane::backend::cpu, {2, 16384}, output);
	run("CPU: 4 workers, grain 4096", iane::backend::cpu, {4, 4096}, output);

#if defined(__CUDACC__)
	int devices = 0;
	const auto status = cudaGetDeviceCount(&devices);
	if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
		(status == cudaSuccess && devices == 0)) {
		std::cout << "CUDA skipped: no usable device\n";
		return;
	}
	iane::detail::check_cuda(status, "Enumerate CUDA devices");
	run("CUDA", iane::backend::cuda, {}, output);
#else
	std::cout << "CUDA skipped: CPU-only build\n";
#endif
}

} // namespace

int main(int aArgc, char** aArgv)
{
	if (aArgc != 1) {
		std::cout << "Usage: execution_policy_benchmark [--help]\n";
		return aArgc == 2 && std::string_view(aArgv[1]) == "--help" ? 0 : 1;
	}
	try {
		run_benchmark();
		return 0;
	} catch (const std::exception& aError) {
		std::cerr << aError.what() << '\n';
		return 1;
	}
}
