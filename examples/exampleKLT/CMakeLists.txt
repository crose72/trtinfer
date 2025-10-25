set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

find_package(vpi 3.2 REQUIRED)

file(GLOB KLT_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB YOLO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../YOLOv8/*.cpp")

add_executable(exampleKLT ${KLT_EXAMPLE_SOURCES} ${YOLO_SOURCES})

target_compile_options(exampleKLT PRIVATE -g3 -O0 -fno-omit-frame-pointer)

target_include_directories(exampleKLT PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../YOLOv8
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include)

target_link_libraries(exampleKLT
    vpi
    ${OpenCV_LIBS}
    cudart
    nvinfer
    trtinfer)

cmake_minimum_required(VERSION 3.5)

project(vpi_sample_06_klt_tracker)

set(CMAKE_CXX_STANDARD 17)