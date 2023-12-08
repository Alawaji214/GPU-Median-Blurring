cmake_minimum_required( VERSION 3.6 )

# Create Project
project( MedianFilter LANGUAGES CUDA CXX)

find_package( OpenCV REQUIRED )

include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    # directory of header files
    find_package(CUDAToolkit)
    enable_language(CUDA)

    include_directories(cuda_median INTERFACE cuda/include)
    add_library(cuda_median cuda/median.cu cuda/common.cu)

    add_executable( video_cuda_median cuda/video_median.cu)
    target_link_libraries( video_cuda_median cuda_median)

    add_executable( image_cuda_median cuda/image_median.cu)
    target_link_libraries( image_cuda_median cuda_median)

    target_link_libraries( video_cuda_median ${OpenCV_LIBS} )
    target_link_libraries( image_cuda_median ${OpenCV_LIBS} )
endif()

include_directories(MedianFilter INTERFACE include)

# Add the source code
add_executable( median median_filter.cpp )

# Copy the refrence images to the binary folder
file(COPY ../resources DESTINATION ${CMAKE_BINARY_DIR})

# link with OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( median ${OpenCV_LIBS} )
