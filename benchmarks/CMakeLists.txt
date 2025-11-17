find_package(benchmark REQUIRED)

add_executable(bench_immersion immersion.cpp)
target_link_libraries(bench_immersion PRIVATE dt::io benchmark::benchmark_main)