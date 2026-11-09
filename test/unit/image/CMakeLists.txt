# ------------------------------------------------------------
# image module unit tests
# ------------------------------------------------------------

add_executable(test_host_image
	test_host_image.cpp
)

target_link_libraries(test_host_image PRIVATE
	iane
	Boost::unit_test_framework
)

add_test(NAME test_host_image COMMAND test_host_image)

if(IANE_ENABLE_CUDA)
	add_executable(test_managed_image test_managed_image.cu)
	target_link_libraries(test_managed_image PRIVATE iane Boost::unit_test_framework)
	add_test(NAME managed_image_cuda COMMAND test_managed_image)
	set_tests_properties(managed_image_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()
