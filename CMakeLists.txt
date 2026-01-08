cmake_minimum_required(VERSION 3.23)
project(distance_transform_benchmark C CXX CUDA)

set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr --extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES 75 80 90)

option(BUILD_TESTING "Build the tests of the package" OFF)
option(BUILD_BINARIES "Build the binaries of the project" OFF)
option(BUILD_BENCHMARKS "Build the benchmarks" OFF)

find_package(CUDAToolkit REQUIRED)

include(FetchContent)

add_subdirectory(dt)
if (BUILD_BINARIES)
    add_subdirectory(bin)
endif(BUILD_BINARIES)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTING)
if (BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif(BUILD_BENCHMARKS)