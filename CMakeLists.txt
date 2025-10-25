cmake_minimum_required(VERSION 3.25)
project(TensorRT_Inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Release)    
set(BUILD_EXAMPLES OFF)

# Set CMAKE_MODULE_PATH if you have custom Find*.cmake files
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# --- TensorRT configuration ---
set(TensorRT_ROOT /opt/tensorrt CACHE PATH "Path to TensorRT installation")
set(TensorRT_INCLUDE_DIRS ${TensorRT_ROOT}/include)
set(TensorRT_LIBRARY_DIRS ${TensorRT_ROOT}/lib)

# Tell CMake where to search for TensorRT libs
link_directories(${TensorRT_LIBRARY_DIRS})

# Optional: explicit libs if you want to list them
set(TensorRT_LIBS
    ${TensorRT_LIBRARY_DIRS}/libnvinfer.so
    ${TensorRT_LIBRARY_DIRS}/libnvinfer_plugin.so
    ${TensorRT_LIBRARY_DIRS}/libnvonnxparser.so
    ${TensorRT_LIBRARY_DIRS}/libnvparsers.so
)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TensorRT REQUIRED)
find_package(fmt REQUIRED)

# Source files from src/
file(GLOB TENSORRT_INFER_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)

# Build the static library
add_library(trtinfer STATIC ${TENSORRT_INFER_SOURCES})

target_include_directories(trtinfer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
)

target_link_libraries(trtinfer
    ${OpenCV_LIBS}
    cudart
    nvinfer
    fmt::fmt
    )

set_target_properties(trtinfer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Build examples
if (BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()