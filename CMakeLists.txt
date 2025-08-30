cmake_minimum_required(VERSION 3.25)
project(TensorRT_Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CMAKE_MODULE_PATH if you have custom Find*.cmake files
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(fmt REQUIRED)

# Source files from src/
file(GLOB TENSORRT_INFER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

# Build the static library
add_library(tensorrtinfer STATIC ${TENSORRT_INFER_SOURCES})

target_include_directories(tensorrtinfer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
)

target_link_libraries(tensorrtinfer
    ${OpenCV_LIBS}
    cudart
    nvinfer
    fmt::fmt
    )

# Build examples
add_subdirectory(examples)