FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.8.3
)

set(BENCHMARK_ENABLE_TESTING OFF)  # Disable benchmark's own tests
FetchContent_MakeAvailable(benchmark)

add_executable(bench_distance_transform distance_transform.cpp)
target_link_libraries(bench_distance_transform PRIVATE dt::core benchmark::benchmark_main)