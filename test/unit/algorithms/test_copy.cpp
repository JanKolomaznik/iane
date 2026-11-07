#define BOOST_TEST_MODULE iane_copy_cpu
#include <boost/test/included/unit_test.hpp>

#include "copy_cases.hpp"

#include <iane/concepts.hpp>
#include <iane/view/device_image_view.hpp>

namespace {

using host_image_t = iane::host_image<int, 3>;
using host_view_t = host_image_t::view_t;
using const_host_view_t = host_image_t::const_view_t;
using device_view_t = iane::device_image_view<int, 3>;
using generated_view_t = meta_tests::host_pattern<host_view_t>;

static_assert(iane::concepts::WritableImage<host_image_t>);
static_assert(iane::concepts::ReadableView<generated_view_t>);
static_assert(!iane::concepts::MemoryView<generated_view_t>);
static_assert(!iane::concepts::WritableView<const_host_view_t>);
static_assert(iane::concepts::TransformViews<const_host_view_t, host_view_t>);
static_assert(!iane::concepts::TransformViews<const_host_view_t, device_view_t>);
static_assert(iane::concepts::TransferViews<const_host_view_t, device_view_t>);
static_assert(!iane::concepts::TransferViews<device_view_t, const_host_view_t>);
static_assert(!iane::concepts::MatchingViewRank<host_view_t, iane::host_image<int, 2>::view_t>);

} // namespace

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_copy<2>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_copy<3>(iane::backend::cpu); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_copy<4>(iane::backend::cpu); }

template<class TLeft, class TRight>
concept can_zip = requires(TLeft aLeft, TRight aRight) {
	iane::zip_view(aLeft, aRight, meta_tests::add_pixels{});
};

using host_expression = decltype(iane::constant_view(const_host_view_t{}, 1));
using vector_expression = decltype(iane::constant_view(const_host_view_t{}, iane::math::vec2i{}));
static_assert(!iane::concepts::WritableView<vector_expression>);
using device_expression = decltype(iane::constant_view(device_view_t{}, 1));
static_assert(can_zip<host_expression, const_host_view_t>);
static_assert(!can_zip<host_expression, device_expression>);
static_assert(!can_zip<host_expression, iane::const_host_image_view<int, 2>>);

BOOST_AUTO_TEST_CASE(procedural_copy)
{
	meta_tests::check_procedural_copy(iane::backend::cpu);
}

BOOST_AUTO_TEST_CASE(procedural_views_are_lazy_and_check_zip_extents)
{
	iane::host_image<int, 1> input(3);
	auto mapped = iane::map_view(input.const_view(), meta_tests::double_pixel{});
	input.pointer()[1] = 4;
	BOOST_TEST(mapped[1] == 8);
	input.pointer()[1] = 9;
	BOOST_TEST(mapped[1] == 18);

	iane::host_image<int, 1> short_input(2);
	auto constant = iane::constant_view(short_input.const_view(), 1);
	BOOST_CHECK_THROW(iane::zip_view(mapped, constant, meta_tests::add_pixels{}), std::invalid_argument);
}

BOOST_AUTO_TEST_CASE(view_algebra)
{
	meta_tests::check_view_algebra(iane::backend::cpu);
}

template<class TLeft, class TRight>
concept can_add_views = requires(TLeft aLeft, TRight aRight) { aLeft + aRight; };

static_assert(can_add_views<host_expression, const_host_view_t>);
static_assert(!can_add_views<host_expression, device_expression>);

BOOST_AUTO_TEST_CASE(view_algebra_checks_extents)
{
	iane::host_image<int, 1> input(3);
	iane::host_image<int, 1> short_input(2);
	BOOST_CHECK_THROW(input.const_view() + short_input.const_view(), std::invalid_argument);
}
