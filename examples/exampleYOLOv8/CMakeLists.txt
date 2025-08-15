set(CMAKE_BUILD_TYPE Debug)                 # or pass -DCMAKE_BUILD_TYPE=Debug

add_executable(exampleYOLOv8 exampleYOLOv8.cpp)

target_compile_options(exampleYOLOv8 PRIVATE -g3 -O0 -fno-omit-frame-pointer)
# Optional (great for STL bounds checks; slower):
#target_compile_definitions(exampleYOLOv8 PRIVATE _GLIBCXX_ASSERTIONS _GLIBCXX_DEBUG)

target_include_directories(exampleYOLOv8 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(exampleYOLOv8
    ${OpenCV_LIBS}
    cudart
    nvinfer
    tensorrtinfer
    )