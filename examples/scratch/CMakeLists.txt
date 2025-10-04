set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

file(GLOB SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB YOLO_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../YOLOv8/*.cpp")

add_executable(testBatchImages ${SOURCES} ${YOLO_SOURCES})

target_compile_options(testBatchImages PRIVATE -g3 -O0 -fno-omit-frame-pointer)

target_include_directories(testBatchImages PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../YOLOv8
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(testBatchImages
    ${OpenCV_LIBS}
    cudart
    nvinfer
    trtinfer
    )