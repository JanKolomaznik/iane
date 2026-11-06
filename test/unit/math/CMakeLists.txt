# ------------------------------------------------------------
# Vector module unit tests
# ------------------------------------------------------------

add_executable(test_vector
	test_vector.cpp
)

target_link_libraries(test_vector PRIVATE
	iane
	Boost::unit_test_framework
)

add_test(NAME vector_tests COMMAND test_vector)
