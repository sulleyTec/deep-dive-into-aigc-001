cmake_minimum_required(VERSION 3.10)  # set minimum version of cmake required

project(experiments LANGUAGES CXX CUDA)  # set project name
set(CMAKE_BUILD_TYPE Debug)  # enable debug symbols for gdb debugging
#set(CMAKE_BUILD_TYPE Release)  # enable debug symbols for gdb debugging
set(CMAKE_CXX_STANDARD 11)  # set C++ standard to 11
set(CMAKE_CUDA_ARCHITECTURES 86)

# Use find_package to locate the OpenCV library
find_package(CUDA REQUIRED)
find_library(CUBLAS_LIBRARIES cublas HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

# Check if CUBLAS library is found
if (CUBLAS_LIBRARIES)
    message("Found CUBLAS library: ${CUBLAS_LIBRARIES}")
else()
    message(FATAL_ERROR "CUBLAS library not found")
endif()

file(GLOB_RECURSE SRC 
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/kernel/*.cu
    ${PROJECT_SOURCE_DIR}/lib/*.cpp)

set(PRO_H ${PROJECT_SOURCE_DIR}/include)

# set include directory
include_directories(PUBLIC ${PRO_H} ${CUDA_INCLUDE_DIRS})

# create an executable target for main.cpp, linking with sql library
add_executable(test ${SRC})
target_link_libraries(test PUBLIC ${CUDA_LIBRARIES} ${CUBLAS_LIBRARIES})


