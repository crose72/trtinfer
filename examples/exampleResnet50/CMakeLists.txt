set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

add_executable(exampleResnet50_v2 exampleResnet50_v2.cpp ResNet50.cpp)

target_compile_options(exampleResnet50_v2 PRIVATE -g3 -O0 -fno-omit-frame-pointer)

target_include_directories(exampleResnet50_v2 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(exampleResnet50_v2
    ${OpenCV_LIBS}
    cudart
    nvinfer
    trtinfer
    )