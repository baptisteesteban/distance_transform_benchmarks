find_package(benchmark REQUIRED)

add_executable(bench_distance_transform distance_transform.cpp)
target_link_libraries(bench_distance_transform PRIVATE dt::core benchmark::benchmark_main)