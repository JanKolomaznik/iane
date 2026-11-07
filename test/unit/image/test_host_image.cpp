#define BOOST_TEST_MODULE iane_host_image_tests
#include <boost/test/included/unit_test.hpp>

#include <iane/image/image.hpp>
#include <iane/resource/host/resource_host.hpp>
#include <iane/math/vector.hpp>	// for dim_traits and simple_vector utilities
#include <iane/view/make_view.hpp>

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

BOOST_AUTO_TEST_CASE(packed_buffer_views)
{
	int pixels[]{1, 2, 3, 4, 5, 6};
	auto writable = host::make_view<int, 2>(pixels, coord2{3, 2});
	auto readable = host::make_const_view<int, 2>(pixels, coord2{3, 2});
	const coord2 last{2, 1};
	BOOST_TEST(readable[last] == 6);
	writable[last] = 19;
	BOOST_TEST(pixels[5] == 19);
	const auto line = host::make_const_view<int, 1>(pixels, 6);
	BOOST_TEST(line[4] == 5);
}

BOOST_AUTO_TEST_CASE(crop_and_slice_preserve_parent_strides)
{
	int pixels[]{1, 2, 3, 4, 5, 6};
	auto parent = host::make_view<int, 2>(pixels, coord2{3, 2});
	const coord2 corner{1, 0};
	const coord2 size{2, 2};
	auto crop = parent.subview(corner, size);
	auto const_crop = host::make_const_view<int, 2>(pixels, coord2{3, 2}).subview(corner, size);
	const coord2 second_row{0, 1};
	BOOST_TEST(const_crop[second_row] == 5);
	auto column = crop.slice<0>(0);
	auto row = crop.slice<1>(1);
	BOOST_TEST(column.dimensions() == 2);
	BOOST_TEST(column.strides() == 3 * sizeof(int));
	BOOST_TEST(column[0] == 2);
	BOOST_TEST(column[1] == 5);
	BOOST_TEST(row[1] == 6);
	row[0] = 23;
	BOOST_TEST(column[1] == 23);
}

BOOST_AUTO_TEST_CASE(slice_3d_preserves_axis_order)
{
	int pixels[24]{};
	using coord3 = math::dim_traits<3>::coord_t;
	auto volume = host::make_view<int, 3>(pixels, coord3{2, 3, 4});
	auto slice = volume.slice<1>(1);
	BOOST_TEST(slice.dimensions()[0] == 2);
	BOOST_TEST(slice.dimensions()[1] == 4);
	const coord2 position{1, 3};
	slice[position] = 37;
	BOOST_TEST(pixels[21] == 37);
}
