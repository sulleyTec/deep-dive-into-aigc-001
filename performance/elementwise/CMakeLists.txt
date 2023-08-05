cmake_minimum_required(VERSION 3.12)
project(elementwise LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 86)

set(CMAKE_BUILD_TYPE Debug)  # enable debug symbols for gdb debugging

SET(CUDA_HEAD "/usr/local/cuda-11.4/targets/x86_64-linux/include/")
SET(PRO_H ${PROJECT_SOURCE_DIR}/include)
SET(KERNEL ${PROJECT_SOURCE_DIR}/kernel)

include_directories(${CUDA_HEAD} ${PRO_H})

set(PY_SRC_FILES src/py.cpp ${KERNEL}/elementwise.cu)
set(SRC_FILES src/main.cpp ${KERNEL}/elementwise.cu)

add_subdirectory("./third_part/pybind11")
pybind11_add_module(libvecmul ${PY_SRC_FILES})

add_executable(elementwise ${SRC_FILES})

set_target_properties(elementwise PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin"
)

set_target_properties(libvecmul PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin"
)

