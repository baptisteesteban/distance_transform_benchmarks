#include <dt/border.hpp>
#include <dt/immersion.hpp>
#include <dt/imread.hpp>
#include <dt/level_lines_distance_transform.hpp>
#include <dt/propagation.hpp>
#include <dt/transfert.hpp>

#include <benchmark/benchmark.h>

static constexpr const char* INPUT_FILENAME = "/data/condat/gray_png/IM001.png";

static void BM_Propagation(benchmark::State& st)
{
  const auto img    = dt::add_median_border(dt::imread<std::uint8_t>(INPUT_FILENAME));
  const auto [m, M] = dt::immersion(img);
  for (auto _ : st)
    dt::propagation<std::uint32_t>(m, M);
}

static void BM_DistanceTransformFG(benchmark::State& st)
{
  const auto _img   = dt::add_median_border(dt::imread<std::uint8_t>(INPUT_FILENAME));
  const auto img    = dt::host_to_device(_img);
  const auto [m, M] = dt::immersion_gpu(img);
  for (auto _ : st)
    dt::level_lines_distance_transform_fg_gpu(m, M);
}

BENCHMARK(BM_Propagation);
BENCHMARK(BM_DistanceTransformFG);

int main(int argc, char* argv[])
{
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}