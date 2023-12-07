cmake_minimum_required( VERSION 3.6 )

# Create Project
project( MedianFilter LANGUAGES CUDA CXX)

find_package( OpenCV REQUIRED )
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 61)

include(CheckLanguage)
check_language(CUDA)

include_directories(MedianFilter INTERFACE include)

# Add the source code
add_executable( median median_filter.cpp )

add_executable( cuda_median cuda/median.cu)


# Copy the refrence images to the binary folder
file(COPY ../resources DESTINATION ${CMAKE_BINARY_DIR})

# link with OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( median ${OpenCV_LIBS} )

target_link_libraries( cuda_median ${OpenCV_LIBS} )