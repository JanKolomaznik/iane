#define BOOST_TEST_MODULE iane_host_image_tests
#include <boost/test/included/unit_test.hpp>

#include <iane/image/image.hpp>
#include <iane/algorithms/convolution.hpp>
#include <iane/resource/host/resource_host.hpp>
#include <iane/math/vector.hpp>	// for dim_traits and simple_vector utilities

using namespace iane;

// -----------------------------------------------------------------------------
// Helper aliases for the tests
// -----------------------------------------------------------------------------
template<typename TElement, int tDim>
using host_img = iane::host_image<TElement, tDim>;

using coord2 = iane::math::dim_traits<2>::coord_t;

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------

BOOST_AUTO_TEST_CASE(construction_and_dimensions_2d)
{
	// 2D host image 10x20
	host_img<int, 2> img(10, 20);

	auto dims = img.dimensions();

	BOOST_TEST(dims[0] == 10);
	BOOST_TEST(dims[1] == 20);

	// Stride test (expected: contiguous, element-major, converted to bytes)
	auto strides = img.strides();

	BOOST_TEST(strides[0] == sizeof(int));				// element stride
	BOOST_TEST(strides[1] == sizeof(int) * 10);		// row stride
}

BOOST_AUTO_TEST_CASE(write_and_read_pixels)
{
	host_img<float, 2> img(4, 3);

	auto v = img.view();

	// Write via view
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 4; ++x) {
			v[coord2{x, y}] = x + y * 10.0f;
		}
	}

	// Read back
	for (int y = 0; y < 3; ++y) {
		for (int x = 0; x < 4; ++x) {
			BOOST_TEST(v[(coord2{x, y})] == x + y * 10.0f);
		}
	}
}

BOOST_AUTO_TEST_CASE(const_view_read_only)
{
	host_img<int, 2> img(3, 2);

	auto v = img.view();

	// Write something
	v[coord2{0, 0}] = 5;
	v[coord2{1, 0}] = 6;
	v[coord2{2, 0}] = 7;

	// const view test
	const auto cv = img.const_view();

	BOOST_TEST(cv[(coord2{0, 0})] == 5);
	BOOST_TEST(cv[(coord2{1, 0})] == 6);
	BOOST_TEST(cv[(coord2{2, 0})] == 7);

	// compile-time: cv(...) should be read-only
	// (no test needed — if assignment compiles, it's wrong)
}

BOOST_AUTO_TEST_CASE(resize_reallocates_memory)
{
	host_img<int, 2> img(2, 2);

	auto old_ptr = img.pointer();

	img.resize(host_img<int, 2>::extents_t{4, 4});

	auto new_ptr = img.pointer();

	BOOST_CHECK(old_ptr != new_ptr);	// should be different memory
	BOOST_TEST(img.dimensions()[0] == 4);
	BOOST_TEST(img.dimensions()[1] == 4);
}

BOOST_AUTO_TEST_CASE(meta_algorithms_on_host_views)
{
	host_img<int, 2> source(3, 1);
	host_img<int, 2> result(3, 1);

	for_each_position(source.view(), [](int& aValue, const coord2& aPosition) {
		aValue = aPosition[0] + 1;
	}).wait();
	for_each(source.view(), [](int& aValue) { aValue *= 2; }).wait();
	transform(source.const_view(), result.view(), [](int aValue) { return aValue + 1; }).wait();
	BOOST_TEST(result.pointer()[2] == 7);

	transform_position(source.const_view(), result.view(), [](int aValue, const coord2& aPosition) {
		return aValue - aPosition[0];
	}).wait();
	BOOST_TEST(result.pointer()[2] == 4);

	int left_samples = 0;
	for_each_locator(source.const_view(), [&](const concepts::Locator auto& aLocator) {
		left_samples += aLocator[coord2{-1, 0}];
	}, border_mode::zero).wait();
	BOOST_TEST(left_samples == 6);
}

BOOST_AUTO_TEST_CASE(convolution_flips_kernel_and_respects_crop_boundary)
{
	host_img<std::uint8_t, 2> source(4, 2);
	host_img<std::uint8_t, 2> result(2, 2);
	host_img<float, 2> kernel(3, 1);
	for_each_position(source.view(), [](std::uint8_t& aValue, const coord2& aPosition) {
		aValue = static_cast<std::uint8_t>(10 * aPosition[0] + 100 * aPosition[1]);
	}).wait();
	kernel.pointer()[0] = 1.0f;
	kernel.pointer()[1] = 0.0f;
	kernel.pointer()[2] = 0.0f;

	const auto crop = source.const_view().subview(coord2{1, 0}, coord2{2, 2});
	convolve(crop, kernel.const_view(), result.view(), {border_mode::clamp}).wait();
	BOOST_TEST(+result.pointer()[0] == 20);
	BOOST_TEST(+result.pointer()[1] == 20);
	BOOST_TEST(+result.pointer()[2] == 120);
	BOOST_TEST(+result.pointer()[3] == 120);

	convolve(crop, kernel.const_view(), result.view(), {border_mode::zero}).wait();
	BOOST_TEST(+result.pointer()[0] == 20);
	BOOST_TEST(+result.pointer()[1] == 0);
	BOOST_TEST(+result.pointer()[2] == 120);
	BOOST_TEST(+result.pointer()[3] == 0);
}
