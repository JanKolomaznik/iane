#define BOOST_TEST_MODULE IANE_VectorTests
#include <boost/test/unit_test.hpp>

#include <iane/math/vector.hpp>
#include <iane/math/vector_ops.hpp>
#include <iane/math/vector_util.hpp>
#include <iane/math/coord_traits.hpp>

using namespace iane::math;

//------------------------------------------------------------
// Helpers
//------------------------------------------------------------

template <typename T, int N>
static bool approx_equal(const vector<T, N> &a, const vector<T, N> &b, T eps = static_cast<T>(1e-5))
{
	for (int i = 0; i < N; ++i) {
		if (std::abs(a[i] - b[i]) > eps) {
			return false;
		}
	}
	return true;
}

//------------------------------------------------------------
// Tests
//------------------------------------------------------------

BOOST_AUTO_TEST_CASE(default_constructor)
{
	vector<int, 3> v;
	BOOST_TEST(v[0] == 0);
	BOOST_TEST(v[1] == 0);
	BOOST_TEST(v[2] == 0);
}

BOOST_AUTO_TEST_CASE(fill_constructor)
{
	vector<float, 4> v(2.5f, std::true_type{});
	for (int i = 0; i < 4; ++i) {
		BOOST_TEST(v[i] == 2.5f);
	}
}

BOOST_AUTO_TEST_CASE(initializer_constructor)
{
	vector<int, 3> v{1, 2, 3};
	BOOST_TEST(v[0] == 1);
	BOOST_TEST(v[1] == 2);
	BOOST_TEST(v[2] == 3);
}

BOOST_AUTO_TEST_CASE(type_conversion_constructor)
{
	vector<int, 3> v1{1, 2, 3};
	vector<float, 3> v2(v1);

	BOOST_TEST(v2[0] == 1.0f);
	BOOST_TEST(v2[1] == 2.0f);
	BOOST_TEST(v2[2] == 3.0f);
}

BOOST_AUTO_TEST_CASE(arithmetic_plus_minus)
{
	vector<int, 3> a{1, 2, 3};
	vector<int, 3> b{4, 5, 6};

	auto c = a + b;
	auto d = b - a;

	BOOST_TEST(c[0] == 5);
	BOOST_TEST(c[1] == 7);
	BOOST_TEST(c[2] == 9);

	BOOST_TEST(d[0] == 3);
	BOOST_TEST(d[1] == 3);
	BOOST_TEST(d[2] == 3);
}

BOOST_AUTO_TEST_CASE(unary_minus)
{
	vector<int, 3> v{1, -2, 3};
	auto r = -v;

	BOOST_TEST(r[0] == -1);
	BOOST_TEST(r[1] == 2);
	BOOST_TEST(r[2] == -3);
}

BOOST_AUTO_TEST_CASE(scalar_multiplication)
{
	vector<int, 3> v{1, 2, 3};

	auto a = 2 * v;
	auto b = v * 3;

	BOOST_TEST(a[0] == 2);
	BOOST_TEST(a[1] == 4);
	BOOST_TEST(a[2] == 6);

	BOOST_TEST(b[0] == 3);
	BOOST_TEST(b[1] == 6);
	BOOST_TEST(b[2] == 9);
}

BOOST_AUTO_TEST_CASE(dot_product)
{
	vector<int, 3> a{1, 2, 3};
	vector<int, 3> b{4, 5, 6};

	BOOST_TEST(dot(a, b) == 32);
}

BOOST_AUTO_TEST_CASE(cross_product)
{
	vector<int, 3> a{1, 0, 0};
	vector<int, 3> b{0, 1, 0};

	auto c = cross(a, b);

	BOOST_TEST(c[0] == 0);
	BOOST_TEST(c[1] == 0);
	BOOST_TEST(c[2] == 1);
}

BOOST_AUTO_TEST_CASE(squared_magnitude_test)
{
	vector<float, 3> v{1.f, 2.f, 2.f};
	BOOST_TEST(squared_magnitude(v) == 9.0f);
}

BOOST_AUTO_TEST_CASE(magnitude_test)
{
	vector<float, 3> v{1.f, 2.f, 2.f};
	BOOST_TEST(std::abs(magnitude(v) - 3.0f) < 1e-6);
}

BOOST_AUTO_TEST_CASE(normalize_test)
{
	vector<float, 3> v{0.f, 3.f, 4.f};
	auto n = normalize(v);

	BOOST_TEST(std::abs(n[0] - 0.0f) < 1e-6);
	BOOST_TEST(std::abs(n[1] - 0.6f) < 1e-6);
	BOOST_TEST(std::abs(n[2] - 0.8f) < 1e-6);
}

BOOST_AUTO_TEST_CASE(min_max_coords)
{
	vector<int, 3> a{1, 4, 3};
	vector<int, 3> b{2, 2, 5};

	auto mn = min_coords(a, b);
	auto mx = max_coords(a, b);

	BOOST_TEST(mn[0] == 1);
	BOOST_TEST(mn[1] == 2);
	BOOST_TEST(mn[2] == 3);

	BOOST_TEST(mx[0] == 2);
	BOOST_TEST(mx[1] == 4);
	BOOST_TEST(mx[2] == 5);
}

BOOST_AUTO_TEST_CASE(round_floor_ceil_test)
{
	vector<float, 3> v{1.2f, 3.8f, -2.5f};

	auto r = round(v);
	auto f = floor(v);
	auto c = ceil(v);

	BOOST_TEST(r[0] == 1);
	BOOST_TEST(r[1] == 4);
	BOOST_TEST(r[2] == -3);

	BOOST_TEST(f[0] == 1);
	BOOST_TEST(f[1] == 3);
	BOOST_TEST(f[2] == -3);

	BOOST_TEST(c[0] == 2);
	BOOST_TEST(c[1] == 4);
	BOOST_TEST(c[2] == -2);
}

BOOST_AUTO_TEST_CASE(insert_remove_dimension_test)
{
	vector<int, 3> a{10, 20, 30};

	auto b = insert_dimension(a, 99, 1);
	BOOST_TEST(b[0] == 10);
	BOOST_TEST(b[1] == 99);
	BOOST_TEST(b[2] == 20);
	BOOST_TEST(b[3] == 30);

	auto c = remove_dimension(b, 2);
	BOOST_TEST(c[0] == 10);
	BOOST_TEST(c[1] == 99);
	BOOST_TEST(c[2] == 30);
}

BOOST_AUTO_TEST_CASE(approx_equal_test)
{
	vector<float, 3> a{1.f, 2.f, 3.f};
	vector<float, 3> b{1.00001f, 2.00002f, 3.00001f};

	BOOST_TEST(approx_equal(a, b, 1e-3f));
}

//------------------------------------------------------------
// Coordinate alias tests
//------------------------------------------------------------

BOOST_AUTO_TEST_CASE(coord_alias_test)
{
	vec3i a{1, 2, 3};
	size2_t b{10, 20};
	Float2 c{1.5f, -0.5f};

	BOOST_TEST(a[1] == 2);
	BOOST_TEST(b[0] == 10);
	BOOST_TEST(c[1] == -0.5f);
}

BOOST_AUTO_TEST_CASE(dim_traits_test)
{
	auto ext2 = dim_traits<2>::create_extents(10, 20);
	BOOST_TEST(ext2[0] == 10);
	BOOST_TEST(ext2[1] == 20);

	auto e1 = dim_traits<1>::create_extents(5);
	BOOST_TEST(e1 == 5);
}
