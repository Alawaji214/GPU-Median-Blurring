cmake_minimum_required( VERSION 3.6 )

# Create Project
project( MedianFilter LANGUAGES CUDA CXX)

find_package( OpenCV REQUIRED )

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
    enable_language(CUDA)
    add_executable( cuda_median cuda/median.cu)
    target_link_libraries( cuda_median ${OpenCV_LIBS} )
endif()

include_directories(MedianFilter INTERFACE include)

# Add the source code
add_executable( median median_filter.cpp )

# Copy the refrence images to the binary folder
file(COPY ../resources DESTINATION ${CMAKE_BINARY_DIR})

# link with OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( median ${OpenCV_LIBS} )
