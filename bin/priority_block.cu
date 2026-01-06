#include <dt/block_priority.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

#include <iostream>

static constexpr int BLOCK_SIZE = 32;

template <typename T>
__global__ static void block_attribute_to_image(const T* attribute, dt::image2d_view<T>& img)
{
  const int bx = blockIdx.x;
  const int by = blockIdx.y;
  const int x  = bx * blockDim.x + threadIdx.x;
  const int y  = by * blockDim.y + threadIdx.y;

  if (x < img.width() && y < img.height())
    img(x, y) = attribute[by * gridDim.x + bx];
}

int main(int argc, char* argv[])
{
  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " mask.png\n";
    return 1;
  }
  const auto mask   = dt::imread<std::uint8_t>(argv[1]);
  const auto d_mask = dt::host_to_device(mask);

  const int grid_width  = (mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int grid_height = (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int N           = grid_width * grid_height;
  std::cout << "Grid with " << N << " blocks\n";
  std::uint8_t*  priorities = nullptr;
  std::uint32_t* cdf        = nullptr;
  cudaMalloc(&priorities, N * sizeof(std::uint8_t)); // Priority of each block
  cudaMalloc(&cdf, 64 * sizeof(std::uint32_t)); // Cumulative distribution function of the priorities in a cuda grid
  dt::compute_block_priorities(d_mask, priorities, cdf);

  dt::image2d<std::uint8_t> d_attr_img(mask.width(), mask.height(), dt::e_memory_kind::GPU);
  {
    dim3 grid_dim(grid_width, grid_height);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    block_attribute_to_image<<<grid_dim, block_dim>>>(priorities, d_attr_img);
  }

  {
    std::uint32_t* h_cdf = (std::uint32_t*)std::malloc(64 * sizeof(std::uint32_t));
    cudaMemcpy(h_cdf, cdf, 64 * sizeof(std::uint32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 63; i++)
      std::cout << std::format("{} ", h_cdf[i]);
    std::cout << std::format("{}\n", h_cdf[63]);
    std::free(h_cdf);
  }

  const auto attr_img = dt::device_to_host(d_attr_img);
  const auto colored  = dt::inferno(dt::normalize<std::uint8_t>(attr_img));
  dt::imsave("priorities.png", colored);

  cudaFree(priorities);
  cudaFree(cdf);

  return 0;
}