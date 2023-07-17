cmake_minimum_required(VERSION 3.12)
project(gfinfer LANGUAGES CXX CUDA)  # set project name

SET(CMAKE_CXX_STANDARD 17)  # set C++ standard to 11
SET(CMAKE_CUDA_ARCHITECTURES 86)

#set(CMAKE_BUILD_TYPE Release)  # enable debug symbols for gdb debugging
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall") # 定义Release编译参数
SET(CMAKE_BUILD_TYPE Debug)  # enable debug symbols for gdb debugging
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb") # 定义Debug编译参数

SET(PROJ_HEAD ${PROJECT_SOURCE_DIR}/include)
SET(CUDA_HEAD "/usr/local/cuda-11.4/targets/x86_64-linux/include/")

FILE(GLOB_RECURSE EXE_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cu)

FILE(GLOB_RECURSE PY_SRC_FILES
    ${PROJECT_SOURCE_DIR}/src/py.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cpp
    ${PROJECT_SOURCE_DIR}/lib/*.cu)

# create a library target
#add_library(mytensor SHARED ${TENSOR})
SET(LOG4CPP "/usr/local/lib/liblog4cplus.so")

include_directories(${PROJ_HEAD} ${CUDA_HEAD})
add_subdirectory("./third_part/pybind11")
pybind11_add_module(libgfinfer ${PY_SRC_FILES})

add_executable(gfinfer ${EXE_SRC_FILES})
target_link_libraries(gfinfer PUBLIC ${LOG4CPP})
target_link_libraries(libgfinfer PUBLIC ${LOG4CPP})


