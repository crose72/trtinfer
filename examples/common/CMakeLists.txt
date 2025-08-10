find_package(CUDA REQUIRED)

add_library(common_utils STATIC
    logger.cpp
    util.cpp
    )

target_include_directories(common_utils PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/local/cuda/include
    )