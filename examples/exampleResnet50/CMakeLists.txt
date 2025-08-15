set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

add_executable(exampleResnet50 exampleResnet50.cpp)

target_compile_options(exampleResnet50 PRIVATE -g3 -O0 -fno-omit-frame-pointer)
# Optional (great for STL bounds checks; slower):
#target_compile_definitions(exampleResnet50 PRIVATE _GLIBCXX_ASSERTIONS _GLIBCXX_DEBUG)

target_include_directories(exampleResnet50 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(exampleResnet50
    ${OpenCV_LIBS}
    cudart
    nvinfer
    tensorrtinfer
    )