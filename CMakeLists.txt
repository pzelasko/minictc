cmake_minimum_required(VERSION 3.26)
project(3crowns)

## Check if using Apple Clang
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
	# This is needed to work around Apple clang not providing openmp support out of the box.
	# If it's giving an error, just comment it out... find_package() apparently fails here despite the hints.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xclang -fopenmp")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L/opt/homebrew/opt/libomp/lib -lomp")
endif()

set(CMAKE_CXX_STANDARD 20)
enable_testing()

add_subdirectory(src)
add_subdirectory(test)
