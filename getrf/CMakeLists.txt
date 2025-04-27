cmake_minimum_required(VERSION 3.25)
set(CMAKE_CUDA_ARCHITECTURES 87 89 90)
project(getrf VERSION 0.1.0 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# 判断是否有 CUDA 环境
find_package(CUDAToolkit)

if (CUDAToolkit_FOUND)
    message(STATUS "CUDA 环境已找到")
else()
    message(STATUS "CUDA 环境未找到")
endif()

# 设置 MAGMA 路径
set(MAGMA_ROOT "/home/ruinique/magma/magma_install")
set(MAGMA_INCLUDE_DIRS "${MAGMA_ROOT}/include")
set(MAGMA_LIBRARY_DIRS "${MAGMA_ROOT}/lib")

# 设置 OpenBLAS 路径
set(OpenBLAS_ROOT "/home/ruinique/usr/local/openblas")
set(OpenBLAS_INCLUDE_DIRS "${OpenBLAS_ROOT}/include")
set(OpenBLAS_LIBRARY_DIRS "${OpenBLAS_ROOT}/lib")

find_package(OpenMP REQUIRED)

# 收集源文件
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")

file(GLOB_RECURSE TEST_SOURCES "test/*.cpp" "test/*.cu")

include_directories(
    ${MAGMA_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS} 
    ${CMAKE_SOURCE_DIR}/src
)

# 添加库路径
link_directories(${MAGMA_LIBRARY_DIRS}
    ${OpenBLAS_LIBRARY_DIRS}
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接 MAGMA 库
target_link_libraries(${PROJECT_NAME} 
    magma
    magma_sparse
    openblas
    OpenMP::OpenMP_CXX
    CUDA::cuda_driver 
    CUDA::cudart
    CUDA::cusparse
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
)