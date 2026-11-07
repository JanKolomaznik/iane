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
