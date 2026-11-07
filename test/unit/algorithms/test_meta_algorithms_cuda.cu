#define BOOST_TEST_MODULE iane_cuda_meta_algorithms
#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>

#include "meta_algorithm_cases.hpp"

BOOST_AUTO_TEST_CASE(images_2d) { meta_tests::check_meta_algorithms<2>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_3d) { meta_tests::check_meta_algorithms<3>(iane::backend::cuda); }
BOOST_AUTO_TEST_CASE(images_4d) { meta_tests::check_meta_algorithms<4>(iane::backend::cuda); }

BOOST_AUTO_TEST_CASE(strided_4d_round_trip)
{
	using image_t = iane::host_image<int, 4>;
	using coord_t = image_t::extents_t;
	const coord_t storage_size{4, 3, 3, 3};
	const coord_t crop_size{2, 2, 2, 2};
	const coord_t origin{1, 1, 1, 1};
	image_t source(storage_size);
	image_t output(storage_size);
	iane::device_image<int, 4> device(crop_size);

	iane::for_each_position(source.view(), [](int& aValue, const coord_t& aPosition) {
		aValue = 1 + aPosition[0] + 4 * (aPosition[1] + 3 * (aPosition[2] + 3 * aPosition[3]));
	});
	iane::for_each(output.view(), [](int& aValue) { aValue = -1; });
	iane::copy(source.const_view().subview(origin, crop_size), device.view()).wait();
	iane::copy(device.const_view(), output.view().subview(origin, crop_size)).wait();

	const auto input = source.const_view();
	iane::for_each_position(output.const_view(), [&](int aValue, const coord_t& aPosition) {
		bool inside = true;
		for (int i = 0; i < coord_t::dim; ++i) {
			inside &= aPosition[i] >= origin[i] && aPosition[i] < origin[i] + crop_size[i];
		}
		BOOST_TEST(aValue == (inside ? input[aPosition] : -1));
	});
}

int main(int aArgc, char** aArgv)
{
	int devices = 0;
	const auto status = cudaGetDeviceCount(&devices);
	if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
	    (status == cudaSuccess && devices == 0)) {
		std::cout << "No usable CUDA device\n";
		return 77;
	}
	iane::detail::check_cuda(status, "Enumerate CUDA devices");
	return boost::unit_test::unit_test_main(&init_unit_test, aArgc, aArgv);
}
