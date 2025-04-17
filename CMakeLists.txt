cmake_minimum_required(VERSION 3.25)
project(MagmaBenchmark VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 87 89 90)

# 判断是否有 CUDA 环境
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA 环境已找到")
else()
    message(STATUS "CUDA 环境未找到")
endif()

add_subdirectory(getrf)