#include <dt/image2d.hpp>
#include <dt/imprint.hpp>
#include <dt/imread.hpp>
#include <dt/imsave.hpp>
#include <dt/inferno.hpp>
#include <dt/normalize.hpp>
#include <dt/priority.hpp>
#include <dt/transfert.hpp>

#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>

#include <climits>
#include <cstdlib>
#include <iostream>
#include <vector>

// (Removed unused kernels `mask_ratio` and `mask_priority`.)

__global__ void mask_has_pixel(const dt::image2d<std::uint8_t>& mask, std::uint8_t* out)
{
  const int      bx = blockIdx.x;
  const int      by = blockIdx.y;
  const int      x  = bx * blockDim.x + threadIdx.x;
  const int      y  = by * blockDim.y + threadIdx.y;
  __shared__ int any;
  if (threadIdx.x == 0 && threadIdx.y == 0)
    any = 0;
  __syncthreads();

  if (x < mask.width() && y < mask.height())
  {
    if (mask(x, y) > 0)
      atomicExch(&any, 1);
  }
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0)
    out[by * gridDim.x + bx] = static_cast<std::uint8_t>(any);
}

__global__ void block_min_distance(const int* sx, const int* sy, int nsrc, int* out)
{
  const int bx  = blockIdx.x;
  const int by  = blockIdx.y;
  const int bid = by * gridDim.x + bx;

  int best = INT_MAX;
  for (int i = 0; i < nsrc; ++i)
  {
    const int d = abs(bx - sx[i]) + abs(by - sy[i]);
    if (d < best)
      best = d;
  }
  if (best == INT_MAX)
    best = -1;
  out[bid] = best;
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
    out(x, y) = attribute[bid];
}

// Compute per-block Manhattan distance priorities where blocks that contain
// any masked pixel are considered sources (distance 0).
std::vector<int> compute_block_priorities(const dt::image2d<std::uint8_t>& mask, int block_size)
{
  const auto d_mask = dt::host_to_device(mask);
  dim3       gridDim((mask.width() + block_size - 1) / block_size, (mask.height() + block_size - 1) / block_size);
  dim3       blockDim(block_size, block_size);
  const int  size = gridDim.x * gridDim.y;

  std::uint8_t* d_has = nullptr;
  cudaMalloc(&d_has, size * sizeof(std::uint8_t));
  mask_has_pixel<<<gridDim, blockDim>>>(d_mask, d_has);
  cudaDeviceSynchronize();

  std::vector<std::uint8_t> h_has(size);
  cudaMemcpy(h_has.data(), d_has, size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);

  std::vector<int> sx, sy;
  for (int i = 0; i < size; ++i)
  {
    if (h_has[i])
    {
      sx.push_back(i % gridDim.x);
      sy.push_back(i / gridDim.x);
    }
  }

  if (sx.empty())
  {
    // No sources: return vector filled with -1
    return std::vector<int>(size, -1);
  }

  const int nsrc = static_cast<int>(sx.size());

  // Use Thrust: upload source coordinates to device vectors
  thrust::device_vector<int> d_sx_vec(sx.begin(), sx.end());
  thrust::device_vector<int> d_sy_vec(sy.begin(), sy.end());

  int* d_sx_ptr = thrust::raw_pointer_cast(d_sx_vec.data());
  int* d_sy_ptr = thrust::raw_pointer_cast(d_sy_vec.data());

  // Use Thrust device_vectors to hold source coordinates, then call the
  // existing `block_min_distance` kernel (no extra host->device copies needed).
  int* d_sx_ptr_local = d_sx_ptr;
  int* d_sy_ptr_local = d_sy_ptr;

  int* d_dist = nullptr;
  cudaMalloc(&d_dist, size * sizeof(int));
  block_min_distance<<<gridDim, dim3(1, 1)>>>(d_sx_ptr_local, d_sy_ptr_local, nsrc, d_dist);
  cudaDeviceSynchronize();

  std::vector<int> h_dist(size);
  cudaMemcpy(h_dist.data(), d_dist, size * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_dist);

  cudaFree(d_has);

  return h_dist;
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

  // Compute per-block priorities (Manhattan distance from masked blocks)
  const auto priorities = compute_block_priorities(mask, BLOCK_SIZE);
  if (priorities.empty())
  {
    std::cerr << "No masked blocks found; nothing to compute.\n";
    return 1;
  }

  // Grid for expanding block attributes to image
  dim3      gridDim((mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE, (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3      blockDim(BLOCK_SIZE, BLOCK_SIZE);
  const int size = gridDim.x * gridDim.y;

  // Copy priorities to device and expand to pixel image
  int* d_priorities = nullptr;
  cudaMalloc(&d_priorities, size * sizeof(int));
  cudaMemcpy(d_priorities, priorities.data(), size * sizeof(int), cudaMemcpyHostToDevice);

  dt::image2d<int> d_attr_img(mask.width(), mask.height(), dt::e_memory_kind::GPU);
  block_attribute_to_image<<<gridDim, blockDim>>>(d_priorities, d_attr_img);
  cudaDeviceSynchronize();

  const auto attr_img = dt::device_to_host(d_attr_img);

  const auto normalized = dt::normalize<std::uint8_t>(attr_img);
  const auto colored    = dt::inferno(normalized);
  dt::imsave("block_priority.png", colored);

  cudaFree(d_priorities);

  return 0;
}