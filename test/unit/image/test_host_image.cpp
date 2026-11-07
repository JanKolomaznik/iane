#define BOOST_TEST_MODULE iane_host_image_tests
#include <boost/test/included/unit_test.hpp>

#include <iane/image/image.hpp>
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
