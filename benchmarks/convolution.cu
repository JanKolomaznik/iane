#include <iane/algorithms/convolution.hpp>
#include <iane/execution/run_on.hpp>
#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace {

using pixel_t = std::uint8_t;
using image_t = iane::host_image<pixel_t, 2>;
using kernel_t = iane::host_image<float, 2>;

void check_output(const image_t& aReference, const image_t& aOutput)
{
	const auto size = aReference.dimensions();
	const auto count = static_cast<std::size_t>(size[0]) * size[1];
	if (!std::equal(aReference.pointer(), aReference.pointer() + count, aOutput.pointer())) {
		throw std::runtime_error("Convolution output differs from the serial result");
	}
}

#if defined(__CUDACC__)
void run_cuda(const image_t& aInput, const kernel_t& aKernel,
	const image_t& aReference, image_t& aOutput)
{
	int devices = 0;
	const auto status = cudaGetDeviceCount(&devices);
	if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
	    (status == cudaSuccess && devices == 0)) {
		std::cout << "CUDA linear and texture skipped: no usable device\n";
		return;
	}
	iane::detail::check_cuda(status, "Enumerate CUDA devices");

	iane::device_image<pixel_t, 2> linear;
	iane::device_image<pixel_t, 2> output;
	iane::device_image<float, 2> kernel;
	std::optional<iane::texture_image<pixel_t>> texture;
	iane::detail::stream_owner queue;
	const auto setup = iane::detail::elapsed_ms([&] {
		iane::detail::check_cuda(cudaStreamCreateWithFlags(&queue.stream, cudaStreamNonBlocking),
			"Create benchmark CUDA stream");
		linear.resize(aInput.dimensions());
		output.resize(aOutput.dimensions());
		kernel.resize(aKernel.dimensions());
	});
	const auto texture_setup = iane::detail::elapsed_ms([&] { texture.emplace(aInput.dimensions()); });
	const iane::cuda_policy policy{.stream = queue.stream};
	const auto kernel_upload = iane::detail::elapsed_ms([&] {
		iane::copy(aKernel.const_view(), kernel.view(), policy).wait();
	});
	const auto linear_upload = iane::detail::elapsed_ms([&] {
		iane::copy(aInput.const_view(), linear.view(), policy).wait();
	});
	const auto texture_upload = iane::detail::elapsed_ms([&] {
		iane::copy(aInput.const_view(), *texture, policy).wait();
	});
	std::cout << "cuda setup_ms=" << setup << " kernel_upload_ms=" << kernel_upload
		<< " linear_upload_ms=" << linear_upload << " texture_setup_ms=" << texture_setup
		<< " texture_upload_ms=" << texture_upload << '\n';

	const auto run = [&](const char* aLabel, auto aSource) {
		std::cout << aLabel << '\n';
		iane::detail::run_samples(iane::run_options{.warmup = 1, .runs = 3}, [&] {
			iane::convolve(aSource, kernel.const_view(), output.view(), {}, policy).wait();
		});
		const auto download = iane::detail::elapsed_ms([&] {
			iane::copy(output.const_view(), aOutput.view(), policy).wait();
		});
		std::cout << "download_ms=" << download << '\n';
		check_output(aReference, aOutput);
	};
	run("CUDA linear", linear.const_view());
	run("CUDA texture", texture->const_view());
}
#endif

void run_benchmark()
{
	constexpr int width = 2560;
	constexpr int height = 1920;
	constexpr int kernel_width = 9;
	const image_t::extents_t size{width, height};
	image_t input(size);
	image_t reference(size);
	image_t output(size);
	kernel_t kernel(kernel_t::extents_t{kernel_width, kernel_width});
	for (int i = 0; i < width * height; ++i) {
		input.pointer()[i] = static_cast<pixel_t>((i * 19) ^ (i / width));
	}
	for (int i = 0; i < kernel_width * kernel_width; ++i) {
		kernel.pointer()[i] = static_cast<float>((i * 3) % 7 - 2) / 64.0f;
	}

	const auto run_cpu = [&](const char* aLabel, image_t& aDestination, iane::cpu_policy aPolicy) {
		std::cout << aLabel << '\n';
		iane::detail::run_samples(iane::run_options{.warmup = 1, .runs = 3}, [&] {
			iane::convolve(input.const_view(), kernel.const_view(), aDestination.view(), {}, aPolicy).wait();
		});
	};
	run_cpu("serial CPU", reference, {});
	run_cpu("CPU: 4 workers", output, {.threads = 4});
	check_output(reference, output);
	run_cpu("CPU: 8 workers", output, {.threads = 8});
	check_output(reference, output);

#if defined(__CUDACC__)
	run_cuda(input, kernel, reference, output);
#else
	std::cout << "CUDA linear and texture skipped: CPU-only build\n";
#endif
}

} // namespace

int main(int aArgc, char** aArgv)
{
	if (aArgc != 1) {
		std::cout << "Usage: convolution_benchmark [--help]\n";
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
