#define BOOST_TEST_MODULE iane_managed_image
#include "../algorithms/cuda_test_main.hpp"

#include <iane/algorithms/for_each.hpp>
#include <iane/resource/cuda/managed_image.hpp>

static_assert(iane::concepts::HostView<decltype(std::declval<iane::managed_image<int>&>().host_view())>);
static_assert(iane::concepts::DeviceView<decltype(std::declval<iane::managed_image<int>&>().device_view())>);
static_assert(!iane::concepts::WritableView<decltype(std::declval<const iane::managed_image<int>&>().host_view())>);
static_assert(!iane::concepts::WritableView<decltype(std::declval<const iane::managed_image<int>&>().device_view())>);
static_assert(std::movable<iane::managed_image<int>>);
static_assert(!std::copy_constructible<iane::managed_image<int>>);

BOOST_AUTO_TEST_CASE(host_and_device_share_storage)
{
	int device{};
	int managed{};
	iane::detail::check_cuda(cudaGetDevice(&device), "Get current CUDA device");
	iane::detail::check_cuda(cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, device),
		"Query managed memory support");
	if (!managed) {
		BOOST_TEST_MESSAGE("CUDA device does not support managed memory");
		return;
	}

	using coord_t = iane::math::vec2i;
	iane::managed_image<int, 2> image(coord_t{4, 3});
	const auto host = image.host_view();
	const auto device_view = image.device_view();
	BOOST_CHECK(host.pointer() == device_view.pointer());

	iane::for_each_position(host, [](int& aPixel, coord_t aPosition) {
		aPixel = aPosition[0] + 10 * aPosition[1];
	});
	auto moved = std::move(image);
	const auto crop = moved.device_view().subview(coord_t{1, 1}, coord_t{2, 1});
	iane::for_each(crop, [] IANE_CUDA_DEVICE (int& aPixel) { aPixel *= 2; }).wait();

	const auto result = std::as_const(moved).host_view();
	const coord_t inside{1, 1};
	const coord_t outside{3, 1};
	BOOST_TEST(result[inside] == 22);
	BOOST_TEST(result[outside] == 13);
	BOOST_CHECK(result.pointer() == std::as_const(moved).device_view().pointer());
}

BOOST_AUTO_TEST_CASE(empty_and_negative_extents)
{
	using coord_t = iane::math::vec2i;
	iane::managed_image<int> empty(coord_t{0, 4});
	BOOST_CHECK(empty.host_view().pointer() == nullptr);
	BOOST_CHECK(empty.device_view().pointer() == nullptr);
	BOOST_TEST(empty.dimensions()[1] == 4);
	BOOST_CHECK_THROW(iane::managed_image<int>(coord_t{-1, 4}), std::invalid_argument);
}
