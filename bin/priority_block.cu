#include <dt/image2d.hpp>
#include <dt/imprint.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/priority.hpp>
#include <dt/transfert.hpp>

#include <iostream>

__global__ void mask_ratio(const dt::image2d<std::uint8_t>& mask, float* out)
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ int count;
  __shared__ int sum;
  if (threadIdx.x == 0 && threadIdx.y == 0)
  {
    count = 0;
    sum   = 0;
  }
  __syncthreads();

  if (x < mask.width() && y < mask.height())
  {
    atomicAdd(&count, 1);
    atomicAdd(&sum, static_cast<int>(mask(x, y)));
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0)
    out[blockIdx.y * gridDim.x + blockIdx.x] = static_cast<float>(sum) / static_cast<float>(count);
}

__global__ void mask_priority(std::uint8_t* out)
{
  const int                     bx = blockIdx.x;
  const int                     by = blockIdx.y;
  dt::manhattan_distance_object d(5, 5);
  if (threadIdx.x == 0 && threadIdx.y == 0)
    out[by * gridDim.x + bx] = d.priorityof(bx, by, gridDim.x, gridDim.y, 64);
}

template <typename T>
__global__ void block_attribute_to_image(const T* attribute, dt::image2d<T>& out)
{
  const int bx  = blockIdx.x;
  const int by  = blockIdx.y;
  const int x   = bx * blockDim.x + threadIdx.x;
  const int y   = by * blockDim.y + threadIdx.y;
  const int bid = by * gridDim.x + bx;

  if (x < out.width() && y < out.height())
  {
    out(x, y) = attribute[bid];
  }
}

int main(int argc, char* argv[])
{
  using A                         = std::uint8_t;
  static constexpr int BLOCK_SIZE = 32;

  if (argc < 2)
  {
    std::cerr << "Usage: " << argv[0] << " mask.png\n";
    return 1;
  }
  const auto mask   = dt::imread<std::uint8_t>(argv[1]);
  const auto d_mask = dt::host_to_device(mask);
  dim3       gridDim((mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE, (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3       blockDim(BLOCK_SIZE, BLOCK_SIZE);
  const int  size             = gridDim.x * gridDim.y;
  A*         block_ratio_attr = nullptr;
  cudaMalloc(&block_ratio_attr, size * sizeof(A));

  mask_priority<<<gridDim, blockDim>>>(block_ratio_attr);

  dt::image2d<A> d_attr_img(mask.width(), mask.height(), dt::e_memory_kind::GPU);
  block_attribute_to_image<<<gridDim, blockDim>>>(block_ratio_attr, d_attr_img);
  cudaFree(block_ratio_attr);
  const auto attr_img = dt::device_to_host(d_attr_img);
  imprint(attr_img);
  const auto normalized = dt::normalize<std::uint8_t>(attr_img);
  const auto colored    = dt::inferno(normalized);
  dt::imsave("block_priority.png", colored);

  return 1;
}