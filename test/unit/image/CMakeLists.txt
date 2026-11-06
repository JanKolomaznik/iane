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
