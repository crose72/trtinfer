set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

file(GLOB YOLO_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB YOLO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../YOLOv8/*.cpp")

add_definitions(-DENABLE_BENCHMARKS)

add_executable(exampleYOLOv8 ${YOLO_EXAMPLE_SOURCES} ${YOLO_SOURCES})

target_compile_options(exampleYOLOv8 PRIVATE -g3 -O0 -fno-omit-frame-pointer)

target_include_directories(exampleYOLOv8 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../YOLOv8
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(exampleYOLOv8
    ${OpenCV_LIBS}
    cudart
    nvinfer
    trtinfer
    )