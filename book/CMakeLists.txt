cmake_minimum_required(VERSION 3.24)
project(cmake_and_cuda LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CUDA_ARCHITECTURES 75)

find_package(OpenCV REQUIRED)

add_executable(01.vecAdd 01.vecAdd.cu)
add_executable(02.colorToGray 02.colorToGray.cu)
target_link_libraries(02.colorToGray ${OpenCV_LIBS})
