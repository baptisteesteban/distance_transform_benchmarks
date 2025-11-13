cmake_minimum_required(VERSION 3.23)
project(distance_transform_benchmark CXX)

option(BUILD_TESTING "Build the tests of the package" OFF)

add_subdirectory(dt)
add_subdirectory(bin)
if (BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif(BUILD_TESTING)