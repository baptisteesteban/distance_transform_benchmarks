#pragma once

#include <cmath>
#include <cstdint>

#include "utils.cuh"

namespace dt
{
  __constant__ float local_dist2d[3];

  // Top -> Bottom
  template <bool Forward>
  __device__ int pass_T(const std::uint8_t img[][TILE_SIZE], float D[][TILE_SIZE], int height, float l_eucl,
                        float l_grad)
  {
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;
    const int     size    = std::min<int>((blockIdx.y + 1) * BLOCK_SIZE, height) - blockIdx.y * BLOCK_SIZE;
    const int     start_y = Forward ? 1 : size;
    const int     end_y   = Forward ? 1 + size : 0;
    const int     x       = threadIdx.x + 1;

    int line_changed = 0;

    for (int y = start_y; y != end_y; y += inc)
    {
      float new_dist = D[y][x];

      for (int dx = -1; dx < 2; ++dx)
      {
        const float l_dist   = minus_abs(img[y][x], img[y + dy][x + dx]);
        const float cur_dist = D[y + dy][x + dx] + l_eucl * local_dist2d[dx + 1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);
      }

      if (new_dist < D[y][x])
      {
        D[y][x] = new_dist;

        line_changed |= BLOCK_CHANGED_ANY;
        line_changed |= (y == 1) * BLOCK_CHANGED_TOP;
        line_changed |= (y == TILE_SIZE - 2) * BLOCK_CHANGED_BOTTOM;
        line_changed |= (x == 1) * BLOCK_CHANGED_LEFT;
        line_changed |= (x == TILE_SIZE - 2) * BLOCK_CHANGED_RIGHT;
      }
      __syncthreads();
    }

    return line_changed;
  }

  // Left -> Right
  template <bool Forward>
  __device__ int pass(const std::uint8_t img[][TILE_SIZE], float D[][TILE_SIZE], int width, float l_eucl, float l_grad)
  {
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;
    const int     size    = std::min<int>((blockIdx.x + 1) * BLOCK_SIZE, width) - blockIdx.x * BLOCK_SIZE;
    const int     start_x = Forward ? 1 : size;
    const int     end_x   = Forward ? 1 + size : 0;
    const int     y       = threadIdx.x + 1;

    int line_changed = 0;

    for (int x = start_x; x != end_x; x += inc)
    {
      float new_dist = D[y][x];

      for (int dy = -1; dy < 2; ++dy)
      {
        const float l_dist   = minus_abs(img[y][x], img[y + dy][x + dx]);
        const float cur_dist = D[y + dy][x + dx] + l_eucl * local_dist2d[dy + 1] + l_grad * l_dist;
        new_dist             = std::min(new_dist, cur_dist);
      }

      if (new_dist < D[y][x])
      {
        D[y][x] = new_dist;

        line_changed |= BLOCK_CHANGED_ANY;
        line_changed |= (y == 1) * BLOCK_CHANGED_TOP;
        line_changed |= (y == TILE_SIZE - 2) * BLOCK_CHANGED_BOTTOM;
        line_changed |= (x == 1) * BLOCK_CHANGED_LEFT;
        line_changed |= (x == TILE_SIZE - 2) * BLOCK_CHANGED_RIGHT;
      }

      __syncthreads();
    }

    return line_changed;
  }
} // namespace dt