#include <dt/block_priority.hpp>
#include <dt/image2d.hpp>
#include <dt/imprint.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/transfert.hpp>

#include <thrust/device_vector.h>

#include <iostream>

template <typename T>
__global__ void block_attribute_to_image(const T* attribute, dt::image2d<T>& out)
{
  const int bx  = blockIdx.x;
  const int by  = blockIdx.y;
  const int x   = bx * blockDim.x + threadIdx.x;
  const int y   = by * blockDim.y + threadIdx.y;
  const int bid = by * gridDim.x + bx;

  if (x < out.width() && y < out.height())
    out(x, y) = attribute[bid];
}


int main(int argc, char* argv[])
{
  static constexpr int BLOCK_SIZE = 32;

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " mask.png\n";
    return 1;
  }
  const auto mask = dt::imread<std::uint8_t>(argv[1]);

  // Compute priorities entirely on GPU
  auto d_priorities = dt::compute_block_priorities(mask, BLOCK_SIZE, 64);
  if (d_priorities.empty())
  {
    std::cerr << "No masked blocks found; nothing to compute.\n";
    return 1;
  }

  dim3 gridDim((mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE, (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  // Expand to image directly from GPU device_vector
  dt::image2d<int> d_attr_img(mask.width(), mask.height(), dt::e_memory_kind::GPU);
  block_attribute_to_image<<<gridDim, blockDim>>>(thrust::raw_pointer_cast(d_priorities.data()), d_attr_img);
  cudaDeviceSynchronize();

  const auto attr_img = dt::device_to_host(d_attr_img);

  const auto normalized = dt::normalize<std::uint8_t>(attr_img);
  const auto colored    = dt::inferno(normalized);
  dt::imsave("block_priority.png", colored);

  return 0;
}