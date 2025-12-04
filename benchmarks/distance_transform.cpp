#include <dt/imread.hpp>
#include <dt/transfert.hpp>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include <filesystem>
#include <format>
#include <iostream>

// Image loading

static constexpr const char* IMAGES_DIR = "";

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
    b->Arg(i);
}

// Fixture

class DistanceTransformFixture : public benchmark::Fixture
{
public:
  void SetUp(benchmark::State& state) override
  {
    const int  image_id = state.range(0);
    const auto filename = bench_filenames[image_id].c_str();
    state.SetLabel(filename);
    auto img = dt::imread<std::atomic_uint8_t>(filename);
    m_img    = dt::image2d<std::uint8_t>(img.width(), img.height(), dt::e_memory_kind::GPU);
    dt::host_to_device(img, m_img);
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

private:
  dt::image2d<std::uint8_t> m_img;
};

// Main

int main(int argc, char* argv[])
{
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
}