set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

file(GLOB YOLO_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_definitions(-DENABLE_BENCHMARKS)

add_executable(exampleBatchYOLOv8 ${YOLO_EXAMPLE_SOURCES})

target_compile_options(exampleBatchYOLOv8 PRIVATE -g3 -O0 -fno-omit-frame-pointer)

target_include_directories(exampleBatchYOLOv8 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(exampleBatchYOLOv8
    ${OpenCV_LIBS}
    cudart
    nvinfer
    trtinfer
    )