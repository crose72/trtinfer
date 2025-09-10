file(GLOB RESNET_EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB RESNET_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/../ResNet50/*.cpp")

add_definitions(-DENABLE_BENCHMARKS)

add_executable(exampleResNet50 ${RESNET_EXAMPLE_SOURCES} ${RESNET_SOURCES})

target_compile_options(exampleResNet50 PRIVATE -g3 -O0 -fno-omit-frame-pointer)

target_include_directories(exampleResNet50 PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ResNet50
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    /usr/local/cuda/include
    )

target_link_libraries(exampleResNet50
    ${OpenCV_LIBS}
    cudart
    nvinfer
    trtinfer
    )