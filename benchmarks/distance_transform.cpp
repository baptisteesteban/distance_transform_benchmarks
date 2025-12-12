#include <dt/generalised_distance_transform.hpp>
#include <dt/imread.hpp>
#include <dt/transfert.hpp>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <format>
#include <iostream>

static constexpr auto  TIME_UNIT = benchmark::kMillisecond;
static constexpr float LAMBDAS[] = {0.0, 0.5, 1.0};

// Image loading

static constexpr const char* IMAGES_DIR = "/data/condat/gray_png";

static std::vector<std::filesystem::path> bench_filenames;

static void load_benchmark_images()
{
  if (!bench_filenames.empty())
    return;

  try
  {
    for (const auto& entry : std::filesystem::directory_iterator(IMAGES_DIR))
    {
      bench_filenames.emplace_back(entry.path());
    }
  }
  catch (std::filesystem::filesystem_error& e)
  {
    std::cerr << std::format("Error loading image: {}\n", e.what());
  }
  std::ranges::sort(bench_filenames);
}

static void build_argument(benchmark::internal::Benchmark* b)
{
  load_benchmark_images();
  for (int i = 0; i < bench_filenames.size(); i++)
  {
    for (int j = 0; j < 3; j++)
      b->Args({i, j});
  }
}

// Fixture

class DistanceTransformFixture : public benchmark::Fixture
{
public:
  void SetUp(benchmark::State& state) override
  {
    // Reading image
    const int  image_id = state.range(0);
    const auto filename = bench_filenames[image_id].c_str();
    state.SetLabel(filename);
    auto img = dt::imread<std::uint8_t>(filename);
    m_img    = dt::image2d<std::uint8_t>(img.width(), img.height(), dt::e_memory_kind::GPU);
    dt::host_to_device(img, m_img);

    // Generating dummy mask
    dt::image2d<std::uint8_t> mask(img.width(), img.height());
    mask(img.width() / 2, img.height() / 2) = 0;
    m_mask = dt::image2d<std::uint8_t>(img.width(), img.height(), dt::e_memory_kind::GPU);
    dt::host_to_device(mask, m_mask);

    const int lambda_id      = state.range(1);
    m_lambda                 = LAMBDAS[lambda_id];
    state.counters["lambda"] = m_lambda;
  }

  void run(benchmark::State& state)
  {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    for (auto _ : state)
    {
      float milliseconds = 0;
      cudaEventRecord(start);
      this->exec(state);
      cudaEventRecord(stop);
      cudaEventSynchronize(stop);
      cudaEventElapsedTime(&milliseconds, start, stop);
      state.SetIterationTime(milliseconds / 1000.0);

      if (auto err = cudaGetLastError(); err != cudaSuccess)
      {
        std::cerr << std::format("Failed to run benchmark due to CUDA error: {}\n", cudaGetErrorString(err));
        std::abort();
      }
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(size()));
    state.counters["size"] = size();

    if (auto err = cudaGetLastError(); err != cudaSuccess)
    {
      std::cerr << std::format("Failed to run benchmark due to CUDA error: {}\n", cudaGetErrorString(err));
      std::abort();
    }
  }

  virtual void exec(benchmark::State& state) const = 0;

  int size() const { return m_img.width() * m_img.height(); }

protected:
  dt::image2d<std::uint8_t> m_img;
  dt::image2d<std::uint8_t> m_mask;
  float                     m_lambda;
};

struct BMDistanceTransformGeos : public DistanceTransformFixture
{
  void exec(benchmark::State&) const override { dt::generalised_distance_transform(m_img, m_mask, m_lambda); }
};

struct BMDistanceTransformChessboard : public DistanceTransformFixture
{
  void exec(benchmark::State&) const override
  {
    dt::generalised_distance_transform_chessboard(m_img, m_mask, m_lambda);
  }
};

// Main
BENCHMARK_DEFINE_F(BMDistanceTransformGeos, BMDistanceTransformGeos)(benchmark::State& state)
{
  run(state);
}

BENCHMARK_DEFINE_F(BMDistanceTransformChessboard, BMDistanceTransformChessboard)(benchmark::State& state)
{
  run(state);
}

BENCHMARK_REGISTER_F(BMDistanceTransformGeos, BMDistanceTransformGeos)
    ->Apply(build_argument)
    ->Unit(TIME_UNIT)
    ->UseManualTime()
    ->Name("DistanceTransformGeos");

BENCHMARK_REGISTER_F(BMDistanceTransformChessboard, BMDistanceTransformChessboard)
    ->Apply(build_argument)
    ->Unit(TIME_UNIT)
    ->UseManualTime()
    ->Name("DistanceTransformChessboard");

int main(int argc, char* argv[])
{
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}