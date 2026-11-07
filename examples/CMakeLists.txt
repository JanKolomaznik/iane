find_package(Boost CONFIG REQUIRED COMPONENTS program_options)

add_library(iane_example_io STATIC common/io.cpp)
target_link_libraries(iane_example_io PRIVATE iane)

foreach(example IN ITEMS color_to_bw convolution)
	if(NOT IANE_ENABLE_CUDA)
		set_source_files_properties(${example}/main.cu PROPERTIES LANGUAGE CXX)
	endif()
	add_executable(${example} ${example}/main.cu)
	target_link_libraries(${example} PRIVATE iane iane_example_io Boost::program_options)
endforeach()
