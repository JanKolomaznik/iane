#pragma once

#define BOOST_TEST_NO_MAIN
#define BOOST_TEST_ALTERNATIVE_INIT_API
#include <boost/test/included/unit_test.hpp>

#include <iane/execution/context.hpp>

#include <iostream>

// Allow CTest to skip GPU suites on hosts without an available CUDA device.
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
