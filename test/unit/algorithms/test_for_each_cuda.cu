#define BOOST_TEST_MODULE iane_for_each_cuda
#include "cuda_test_main.hpp"

#include "for_each_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_for_each<2>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_for_each<3>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_for_each<4>(iane::backend::cuda); }

namespace {

// Fill the device output while using a caller-supplied stream.
struct fill_pixel {
	IANE_DECL_HYBRID void operator()(int& aPixel) const { aPixel = 42; }
};

} // namespace

BOOST_AUTO_TEST_CASE(cuda_policy_uses_the_supplied_stream)
{
	iane::host_image<int, 1> output(17);
	iane::run_on(iane::run_options{.backend = iane::backend::cuda}, output.view(),
		[](auto aDestination, auto aExecution) {
			if constexpr (iane::concepts::DeviceView<decltype(aDestination)>) {
				const iane::cuda_policy policy{.stream = aExecution.stream};
				const auto result = iane::for_each(aDestination, fill_pixel{}, policy);
				BOOST_CHECK(result.stream == aExecution.stream);
				return result;
			} else {
				return aExecution;
			}
		});
	for (int i = 0; i < 17; ++i) {
		BOOST_TEST(output.pointer()[i] == 42);
	}
}
