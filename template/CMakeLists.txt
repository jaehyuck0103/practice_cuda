cmake_minimum_required(VERSION 3.17)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_C_COMPILER "gcc-11")
set(CMAKE_CXX_COMPILER "g++-11")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE true)

add_library(kernels kernels/test.cu kernels/test.cuh)
set_target_properties(kernels PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(main main.cpp)
target_link_libraries(main kernels)
