#include <dt/immersion.hpp>
#include <dt/imread.hpp>
#include <dt/transfert.hpp>

#include <benchmark/benchmark.h>

static constexpr const char* INPUT_FILENAME = "/data/condat/gray_png/IM001.png";

static void BM_Immersion_CPU(benchmark::State& st)
{
  const auto img = dt::imread<std::uint8_t>(INPUT_FILENAME);
  for (auto _ : st)
    dt::immersion(img);
}

static void BM_Immersion_GPU_Global(benchmark::State& st)
{
  const auto _img = dt::imread<std::uint8_t>(INPUT_FILENAME);
  const auto img  = dt::host_to_device(_img);
  for (auto _ : st)
    dt::immersion_gpu(img);
}

BENCHMARK(BM_Immersion_CPU);
BENCHMARK(BM_Immersion_GPU_Global);

int main(int argc, char* argv[])
{
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}
