cmake_minimum_required(VERSION 3.24)

project(my_project LANGUAGES C CXX CUDA)

# set(CMAKE_C_COMPILER "gcc-11") set(CMAKE_CXX_COMPILER "g++-11")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_BUILD_TYPE Release)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_VERBOSE_MAKEFILE ON)

# Find Packages
find_package(OpenCV REQUIRED PATHS $ENV{HOME}/Utils/opencv/build
             NO_DEFAULT_PATH)
find_package(CUDAToolkit)
find_package(nvcv_types REQUIRED)
find_package(cvcuda REQUIRED)

include(FetchContent)

# Add executable
add_executable(exec_box_filter box_filter.cpp box_filter.cu)
target_link_libraries(exec_box_filter PRIVATE ${OpenCV_LIBS} CUDA::cudart
                                              nvcv_types cvcuda nppif)
set_target_properties(exec_box_filter PROPERTIES CUDA_ARCHITECTURES native)

add_executable(exec_sort sort.cpp sort.cu)
target_link_libraries(exec_sort PRIVATE ${OpenCV_LIBS} CUDA::cudart)
set_target_properties(exec_sort PROPERTIES CUDA_ARCHITECTURES native)

add_executable(exec_bilateral_filter bilateral_filter.cpp bilateral_filter.cu)
target_link_libraries(exec_bilateral_filter PRIVATE ${OpenCV_LIBS} CUDA::cudart
                                                    nppif)
set_target_properties(exec_bilateral_filter PROPERTIES CUDA_ARCHITECTURES
                                                       native)

add_executable(exec_resize resize.cpp resize.cu)
target_link_libraries(exec_resize PRIVATE ${OpenCV_LIBS} CUDA::cudart nppig)
set_target_properties(exec_resize PROPERTIES CUDA_ARCHITECTURES native)
