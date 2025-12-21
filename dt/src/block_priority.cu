#include <dt/block_priority.hpp>

#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

#include "utils.cuh"

namespace dt
{
  __global__ static void initialize_distance(const image2d_view<std::uint8_t>& mask, std::uint32_t* distance)
  {
    const int bx      = blockIdx.x;
    const int by      = blockIdx.y;
    const int y       = by * blockDim.x + threadIdx.x;
    const int start_x = bx * blockDim.x;
    const int end_x   = std::min<int>(start_x + blockDim.x, mask.width());

    __shared__ int block_has_mask;
    if (threadIdx.x == 0)
      block_has_mask = 0;

    int line_has_mask = 0;
    if (y < mask.height())
    {
      for (int x = start_x; x != end_x; x++)
        line_has_mask |= mask(x, y);
    }
    __syncthreads();

    atomicOr(&block_has_mask, line_has_mask);
    __syncthreads();

    if (threadIdx.x == 0)
      distance[by * gridDim.x + bx] = (block_has_mask == 0) * distance[by * gridDim.x + bx];
  }

  template <bool Forward>
  __global__ static void pass_manathan(std::uint32_t* distance, int width, int height)
  {
    const int     y       = blockIdx.x * blockDim.x + threadIdx.x;
    const int     start_x = Forward ? 1 : width - 2;
    const int     end_x   = Forward ? width : -1;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;

    if (y < height)
    {
      for (int x = start_x; x != end_x; x += inc)
        distance[y * width + x] = std::min(distance[y * width + x + dx] + 1, distance[y * width + x]);
    }
  }

  template <bool Forward>
  __global__ static void pass_T_manathan(std::uint32_t* distance, int width, int height)
  {
    const int     x       = blockIdx.x * blockDim.x + threadIdx.x;
    const int     start_y = Forward ? 1 : height - 2;
    const int     end_y   = Forward ? height : -1;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;

    if (x < width)
    {
      for (int y = start_y; y != end_y; y += inc)
        distance[y * width + x] = std::min(distance[(y + dy) * width + x] + 1, distance[y * width + x]);
    }
  }

  void compute_block_priorities(const dt::image2d_view<std::uint8_t>& mask, std::uint8_t* priorities,
                                std::uint32_t* cdf)
  {
    assert(mask.memory_kind() == dt::e_memory_kind::GPU);

    const int      grid_width  = (mask.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int      grid_height = (mask.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int      N           = grid_width * grid_height;
    std::uint32_t* distance    = nullptr;
    cudaMalloc(&distance, N * sizeof(std::uint32_t));
    // Fill distance with a large value
    {
      auto ptr = thrust::device_pointer_cast(distance);
      thrust::fill(ptr, ptr + N, std::numeric_limits<std::uint32_t>::max() / 2);
    }
    // Initialize the distance (0: block has seed points, large value otherwise)
    {
      dim3 grid_dim(grid_width, grid_height);
      dim3 block_dim(BLOCK_SIZE);
      initialize_distance<<<grid_dim, block_dim>>>(mask, distance);
      cudaDeviceSynchronize();
    }

    // Four passes manathan distance computation
    {
      dim3 grid_dim((grid_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
      dim3 block_dim(BLOCK_SIZE);
      pass_manathan<true><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
      pass_manathan<false><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
    }
    {
      dim3 grid_dim((grid_width + BLOCK_SIZE - 1) / BLOCK_SIZE);
      dim3 block_dim(BLOCK_SIZE);
      pass_T_manathan<true><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
      pass_T_manathan<false><<<grid_dim, block_dim>>>(distance, grid_width, grid_height);
    }
    cudaDeviceSynchronize();

    // Normalization and CDF computation
    {
      const auto ptr            = thrust::device_pointer_cast(distance);
      auto       priorities_ptr = thrust::device_pointer_cast(priorities);
      // Normalization and priority computation from manathan distance
      const auto max_dist = *thrust::max_element(ptr, ptr + N);
      thrust::transform(ptr, ptr + N, priorities_ptr,
                        [max_dist] __device__(const auto& v) { return (static_cast<float>(v) / max_dist) * 63; });
      // CDF
      auto cdf_ptr = thrust::device_pointer_cast(cdf);
      cudaMemset(cdf, 0, 64 * sizeof(std::uint32_t));
      thrust::for_each(priorities_ptr, priorities_ptr + N, [cdf] __device__(const auto v) { atomicAdd(&cdf[v], 1); });
      thrust::inclusive_scan(cdf_ptr, cdf_ptr + 64, cdf_ptr);
    }
    cudaFree(distance);
  }
} // namespace dt