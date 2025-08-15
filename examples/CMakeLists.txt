cmake_minimum_required(VERSION 3.25)
project(TensorRT_Samples)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CMAKE_MODULE_PATH if you have custom Find*.cmake files
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Find packages once at the top level
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)

# Add subdirectories for each example
add_subdirectory(exampleResnet50)