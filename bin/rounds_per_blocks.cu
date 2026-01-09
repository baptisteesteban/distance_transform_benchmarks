#include <dt/generalised_distance_transform.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

#include <cstdlib>
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
  if (argc < 4)
  {
    std::cerr << "Usage: " << argv[0] << " img.png mask.png lmd\n";
    return 1;
  }

  const auto img   = dt::imread<std::uint8_t>(argv[1]);
  const auto d_img = dt::host_to_device(img);

  const float lambda = std::atof(argv[3]);

  const auto mask   = dt::imread<std::uint8_t>(argv[2]);
  const auto d_mask = dt::host_to_device(mask);

  const int grid_width        = (mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int grid_height       = (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const int N                 = grid_width * grid_height;
  int*      rounds_per_blocks = nullptr;
  cudaMallocManaged(&rounds_per_blocks, N * sizeof(int));
  cudaMemset(rounds_per_blocks, 0, N * sizeof(int));

  generalised_distance_transform_blocks(img, mask, lambda, 1e10, rounds_per_blocks);

  dt::image2d<int> d_attr_img(mask.width(), mask.height(), dt::e_memory_kind::GPU);
  {
    dim3 grid_dim(grid_width, grid_height);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    block_attribute_to_image<<<grid_dim, block_dim>>>(rounds_per_blocks, d_attr_img);
  }

  const auto attr_img = dt::device_to_host(d_attr_img);
  const auto colored  = dt::inferno(dt::normalize<std::uint8_t>(attr_img));
  dt::imsave("rounds_per_block.png", colored);

  cudaFree(rounds_per_blocks);

  return 0;
}