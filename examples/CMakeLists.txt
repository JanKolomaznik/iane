find_package(Boost CONFIG REQUIRED COMPONENTS program_options json)

add_library(iane_example_io STATIC common/io.cpp)
target_link_libraries(iane_example_io PRIVATE iane)

add_library(iane_example_cli STATIC common/cli.cpp)
target_link_libraries(iane_example_cli PUBLIC iane Boost::program_options PRIVATE Boost::json)

foreach(example IN ITEMS color_to_bw convolution fractals)
	if(NOT IANE_ENABLE_CUDA)
		set_source_files_properties(${example}/main.cu PROPERTIES LANGUAGE CXX)
	endif()
	add_executable(${example} ${example}/main.cu)
	target_link_libraries(${example} PRIVATE iane_example_cli iane_example_io)
endforeach()
