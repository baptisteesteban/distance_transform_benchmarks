#pragma once

#include <cmath>
#include <cstdint>

#include "utils.cuh"

namespace dt
{
  __constant__ float local_dist2d[3];

  // Left -> Right
  template <bool Forward>
  __device__ bool pass(const std::uint8_t img[][TILE_SIZE], float D[][TILE_SIZE], int width, int height, float l_eucl,
                       float l_grad)
  {
    const int block_start = blockIdx.x * blockDim.x;
    const int block_end   = std::min<int>((blockIdx.x + 1) * blockDim.x, width);
    const int block_width = block_end - block_start;

    constexpr int dx      = Forward ? -1 : 1;
    constexpr int inc     = -1 * dx;
    const int     start_x = Forward ? 1 + (block_start == 0) : block_width - (blockIdx.x == gridDim.x - 1);
    const int     end_x   = Forward ? 1 + block_width : 0;

    const int  ty     = threadIdx.x + 1;
    const int  y      = blockIdx.y * blockDim.x + threadIdx.x;
    const bool active = y < height;

    bool line_changed = false;

    for (int tx = start_x; tx != end_x; tx += inc)
    {
      if (active)
      {
        float       new_dist = D[ty][tx];
        const float l_dist   = minus_abs(img[ty][tx], img[ty][tx + dx]);
        const float cur_dist = D[ty][tx + dx] + l_eucl * local_dist2d[1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);

        if (new_dist < D[ty][tx])
        {
          line_changed = true;
          D[ty][tx]    = new_dist;
        }
      }
      __syncthreads();
    }
    return line_changed;
  }

  // Left -> Right
  template <bool Forward>
  __device__ bool pass_T(const std::uint8_t img[][TILE_SIZE], float D[][TILE_SIZE], int width, int height, float l_eucl,
                         float l_grad)
  {
    const int block_start  = blockIdx.y * blockDim.x;
    const int block_end    = std::min<int>((blockIdx.y + 1) * blockDim.x, height);
    const int block_height = block_end - block_start;

    constexpr int dy      = Forward ? -1 : 1;
    constexpr int inc     = -1 * dy;
    const int     start_y = Forward ? 1 + (block_start == 0) : block_height - (blockIdx.y == gridDim.y - 1);
    const int     end_y   = Forward ? 1 + block_height : 0;

    const int  tx     = threadIdx.x + 1;
    const int  x      = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = x < width;

    bool line_changed = false;

    for (int ty = start_y; ty != end_y; ty += inc)
    {
      if (active)
      {
        float       new_dist = D[ty][tx];
        const float l_dist   = minus_abs(img[ty][tx], img[ty + dy][tx]);
        const float cur_dist = D[ty + dy][tx] + l_eucl * local_dist2d[1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);

        if (new_dist < D[ty][tx])
        {
          line_changed = true;
          D[ty][tx]    = new_dist;
        }
      }
      __syncthreads();
    }
    return line_changed;
  }
} // namespace dt