#define BOOST_TEST_MODULE iane_resample_cuda
#include "cuda_test_main.hpp"

#include "resample_cases.hpp"

#include <iane/resource/cuda/texture_image.hpp>

BOOST_AUTO_TEST_CASE(resize_and_sample_neighborhood)
{
	resample_tests::check_resampling(iane::backend::cuda);
}

BOOST_AUTO_TEST_CASE(resample_volume_into_plane)
{
	resample_tests::check_plane(iane::backend::cuda);
}

BOOST_AUTO_TEST_CASE(resample_point_texture_and_software_nearest)
{
	iane::host_image<int, 2> input(4, 1);
	iane::host_image<int, 2> output(2, 1);
	iane::device_image<int, 2> device_output(2, 1);
	iane::texture_image<int> texture(input.dimensions());
	for (int i = 0; i < 4; ++i) {
		input.pointer()[i] = 10 + i;
	}
	iane::copy(input.const_view(), texture);
	const auto source = texture.const_view();
	iane::resample(source, device_output.view(), resample_tests::scale_x{});
	iane::copy(device_output.const_view(), output.view()).wait();
	BOOST_TEST(output.pointer()[0] == 11);
	BOOST_TEST(output.pointer()[1] == 13);

	iane::for_each(device_output.view(), [] IANE_CUDA_DEVICE (int& aPixel) { aPixel = -1; });
	const auto nearest = iane::make_interpolated_view(source, iane::nearest);
	iane::resample(nearest, device_output.view(), resample_tests::scale_x{});
	iane::copy(device_output.const_view(), output.view()).wait();
	BOOST_TEST(output.pointer()[0] == 11);
	BOOST_TEST(output.pointer()[1] == 13);
}
