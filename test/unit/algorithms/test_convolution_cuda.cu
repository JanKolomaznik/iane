#define BOOST_TEST_MODULE iane_convolution_cuda
#include "cuda_test_main.hpp"

#include "convolution_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_convolution<2>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_convolution<3>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_convolution<4>(iane::backend::cuda); }

static_assert(iane::concepts::ReadableView<iane::texture_image_view<float>>);
static_assert(!iane::concepts::MemoryView<iane::texture_image_view<float>>);
static_assert(!iane::concepts::WritableView<iane::texture_image_view<float>>);

BOOST_AUTO_TEST_CASE(texture_upload_from_strided_host_and_device_views)
{
	using coord_t = iane::math::vec2i;
	const coord_t size{2, 2};
	iane::host_image<int, 2> host(coord_t{4, 3});
	for (int i = 0; i < 12; ++i) {
		host.pointer()[i] = i + 101;
	}
	iane::host_image<int, 2> output(size);
	iane::texture_image<int> texture(size);
	iane::device_image<int, 2> linear(host.dimensions());
	iane::device_image<int, 2> materialized(size);
	iane::detail::stream_owner queue;
	iane::detail::check_cuda(cudaStreamCreateWithFlags(&queue.stream, cudaStreamNonBlocking),
		"Create texture test stream");
	const iane::cuda_policy policy{.stream = queue.stream};

	BOOST_CHECK_THROW(iane::copy(host.const_view(), texture, policy), std::invalid_argument);
	const auto verify = [&](int aOffset) {
		iane::copy(texture.const_view(), materialized.view(), policy);
		iane::copy(materialized.const_view(), output.view(), policy).wait();
		const int expected[]{106, 107, 110, 111};
		for (int i = 0; i < 4; ++i) {
			BOOST_TEST(output.pointer()[i] == expected[i] + aOffset);
		}
	};

	const coord_t origin{1, 1};
	iane::copy(host.const_view().subview(origin, size), texture, policy);
	verify(0);

	for (int i = 0; i < 12; ++i) {
		host.pointer()[i] += 100;
	}
	iane::copy(host.const_view(), linear.view(), policy);
	iane::copy(linear.const_view().subview(origin, size), texture, policy);
	verify(100);
}

BOOST_AUTO_TEST_CASE(texture_convolution_matches_cpu_for_both_borders)
{
	using coord_t = iane::math::vec2i;
	const coord_t size{5, 4};
	const coord_t crop_size{3, 3};
	const coord_t origin{1, 1};
	iane::host_image<float, 2> source(size);
	iane::host_image<float, 2> kernel(coord_t{3, 3});
	iane::host_image<float, 2> reference(crop_size);
	iane::host_image<float, 2> output(crop_size);
	for (int i = 0; i < 20; ++i) {
		source.pointer()[i] = static_cast<float>(i - 6);
	}
	for (int i = 0; i < 9; ++i) {
		kernel.pointer()[i] = static_cast<float>(i - 3) / 8.0f;
	}

	iane::texture_image<float> texture(size);
	iane::device_image<float, 2> device_kernel(kernel.dimensions());
	iane::device_image<float, 2> device_output(crop_size);
	iane::copy(source.const_view(), texture).wait();
	iane::copy(kernel.const_view(), device_kernel.view()).wait();

	for (const auto border : {iane::border_mode::clamp, iane::border_mode::zero}) {
		iane::convolve(source.const_view().subview(origin, crop_size), kernel.const_view(),
			reference.view(), {.border = border});
		iane::convolve(texture.const_view().subview(origin, crop_size), device_kernel.const_view(),
			device_output.view(), {.border = border});
		iane::copy(device_output.const_view(), output.view()).wait();
		for (int i = 0; i < 9; ++i) {
			BOOST_TEST(output.pointer()[i] == reference.pointer()[i]);
		}
	}
}

BOOST_AUTO_TEST_CASE(empty_texture_and_move)
{
	using coord_t = iane::math::vec2i;
	const coord_t size{0, 3};
	iane::texture_image<float> empty(size);
	auto moved = std::move(empty);
	iane::host_image<float, 2> source(size);
	iane::copy(source.const_view(), moved).wait();
	BOOST_TEST(moved.const_view().dimensions()[0] == size[0]);
	BOOST_TEST(moved.const_view().dimensions()[1] == size[1]);
	BOOST_CHECK(moved.array() == nullptr);
}

BOOST_AUTO_TEST_CASE(texture_1d_upload_and_nested_crop)
{
	iane::host_image<int, 1> source(6);
	for (int i = 0; i < 6; ++i) {
		source.view()[i] = 100 + i;
	}
	iane::texture_image<int, 1> texture(4);
	iane::device_image<int, 1> materialized(2);
	iane::host_image<int, 1> output(2);
	iane::copy(source.const_view().subview(1, 4), texture);
	iane::copy(texture.const_view().subview(1, 3).subview(1, 2), materialized.view());
	iane::copy(materialized.const_view(), output.view()).wait();
	BOOST_TEST(output.view()[0] == 103);
	BOOST_TEST(output.view()[1] == 104);
}

BOOST_AUTO_TEST_CASE(texture_3d_upload_from_strided_host_and_device_views)
{
	using coord_t = iane::math::vec3i;
	const coord_t size{3, 2, 3};
	const coord_t crop_size{2, 2, 2};
	const coord_t origin{1, 0, 1};
	iane::texture_image<int, 3> texture(size);
	iane::device_image<int, 3> materialized(crop_size);
	iane::host_image<int, 3> output(crop_size);
	const auto verify = [&](int aOffset) {
		iane::copy(texture.const_view().subview(origin, crop_size), materialized.view());
		iane::copy(materialized.const_view(), output.view()).wait();
		for (int z = 0; z < 2; ++z) {
			for (int y = 0; y < 2; ++y) {
				for (int x = 0; x < 2; ++x) {
					const coord_t position{x, y, z};
					BOOST_TEST(output.view()[position] == (100 * (z + 1) + 10 * y + x + 1 + aOffset));
				}
			}
		}
	};

	// Use a slice stride that is not a multiple of the row pitch.
	int padded[33]{};
	const iane::host_image_view<int, 3> host(padded, size, coord_t{4, 16, 44});
	for (int z = 0; z < 3; ++z) {
		for (int y = 0; y < 2; ++y) {
			for (int x = 0; x < 3; ++x) {
				host[coord_t{x, y, z}] = 100 * z + 10 * y + x;
			}
		}
	}
	iane::copy(host, texture);
	verify(0);

	const coord_t storage_size{5, 4, 4};
	const coord_t storage_origin{1, 1, 1};
	iane::host_image<int, 3> packed(storage_size);
	iane::for_each(packed.view(), [](int& aPixel) { aPixel = -1; });
	auto crop = packed.view().subview(storage_origin, size);
	iane::copy(host, crop);
	iane::for_each(crop, [](int& aPixel) { aPixel += 1000; });
	iane::device_image<int, 3> linear(storage_size);
	iane::copy(packed.const_view(), linear.view());
	iane::copy(linear.const_view().subview(storage_origin, size), texture);
	verify(1000);
}
