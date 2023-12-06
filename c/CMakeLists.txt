cmake_minimum_required( VERSION 3.6 )

# Create Project
project( MedianFilter )

find_package( OpenCV REQUIRED )

# Add the source code
add_executable( median median_filter.cpp )

# Copy the refrence images to the binary folder
file(COPY ../resources DESTINATION ${CMAKE_BINARY_DIR})

# link with OpenCV
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( median ${OpenCV_LIBS} )