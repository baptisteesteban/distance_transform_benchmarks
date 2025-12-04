#pragma once

#include <cstdint>

#include "utils.cuh"

namespace dt
{
  __constant__ float local_dist2d[3];

  // Left -> Right
  template <bool Forward>
  __device__ int pass(const std::uint8_t img[][TILE_SIZE], float D[][TILE_SIZE], int width, int height, float l_eucl,
                      float l_grad, int bx = -1, int by = -1)
  {
    if (bx < 0)
      bx = blockIdx.x;
    if (by < 0)
      by = blockIdx.y;

    const int gx          = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int block_start = bx * BLOCK_SIZE;
    const int block_end   = std::min<int>((bx + 1) * BLOCK_SIZE, width);
    const int block_width = block_end - block_start;

    constexpr int dx      = Forward ? -1 : 1;
    constexpr int inc     = -1 * dx;
    const int     start_x = Forward ? 1 + (block_start == 0) : block_width - (bx == gx - 1);
    const int     end_x   = Forward ? 1 + block_width : 0;

    const int  ty     = threadIdx.x + 1;
    const int  y      = by * BLOCK_SIZE + threadIdx.x;
    const bool active = y < height;

    int line_changed = 0;

    for (int tx = start_x; tx != end_x; tx += inc)
    {
      if (active)
      {
        float new_dist = D[ty][tx];
        for (int dy = -1; dy < 2; dy++)
        {
          const float l_dist   = minus_abs(img[ty][tx], img[ty + dy][tx + dx]);
          const float cur_dist = D[ty + dy][tx + dx] + l_eucl * local_dist2d[dy + 1] + l_grad * l_dist;
          new_dist             = std::min(new_dist, cur_dist);
        }

        if (new_dist < D[ty][tx])
        {
          D[ty][tx] = new_dist;

          line_changed |= BLOCK_CHANGED_ANY;
          line_changed |= (ty == 1) * BLOCK_CHANGED_TOP;
          line_changed |= (ty == TILE_SIZE - 2) * BLOCK_CHANGED_BOTTOM;
          line_changed |= (tx == 1) * BLOCK_CHANGED_LEFT;
          line_changed |= (tx == TILE_SIZE - 2) * BLOCK_CHANGED_RIGHT;
        }
      }
      __syncthreads();
    }
    return line_changed;
  }

  // Left -> Right
  template <bool Forward>
  __device__ int pass_T(const std::uint8_t img[][TILE_SIZE], float D[][TILE_SIZE], int width, int height, float l_eucl,
                        float l_grad, int bx = -1, int by = -1)
  {
    if (bx < 0)
      bx = blockIdx.x;
    if (by < 0)
      by = blockIdx.y;

    const int gy           = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int block_start  = by * BLOCK_SIZE;
    const int block_end    = std::min<int>((by + 1) * BLOCK_SIZE, height);
    const int block_height = block_end - block_start;

    constexpr int dy      = Forward ? -1 : 1;
    constexpr int inc     = -1 * dy;
    const int     start_y = Forward ? 1 + (block_start == 0) : block_height - (by == gy - 1);
    const int     end_y   = Forward ? 1 + block_height : 0;

    const int  tx     = threadIdx.x + 1;
    const int  x      = bx * BLOCK_SIZE + threadIdx.x;
    const bool active = x < width;

    int line_changed = 0;

    for (int ty = start_y; ty != end_y; ty += inc)
    {
      if (active)
      {
        float new_dist = D[ty][tx];
        for (int dx = -1; dx < 2; dx++)
        {
          const float l_dist   = minus_abs(img[ty][tx], img[ty + dy][tx + dx]);
          const float cur_dist = D[ty + dy][tx + dx] + l_eucl * local_dist2d[dx + 1] + l_grad * l_dist;
          new_dist             = std::min(new_dist, cur_dist);
        }

        if (new_dist < D[ty][tx])
        {
          D[ty][tx] = new_dist;

          line_changed |= BLOCK_CHANGED_ANY;
          line_changed |= (ty == 1) * BLOCK_CHANGED_TOP;
          line_changed |= (ty == TILE_SIZE - 2) * BLOCK_CHANGED_BOTTOM;
          line_changed |= (tx == 1) * BLOCK_CHANGED_LEFT;
          line_changed |= (tx == TILE_SIZE - 2) * BLOCK_CHANGED_RIGHT;
        }
      }
      __syncthreads();
    }
    return line_changed;
  }
} // namespace dt