#pragma once

#include <cstdint>

#include "utils.cuh"

namespace dt
{
  static constexpr int BLOCK_SIZE = 32;
  static constexpr int TILE_SIZE  = BLOCK_SIZE + 2; // Halo of 2 to handle border values
  static constexpr int WINDOW     = 1;              // 0 or 1 (if > 1 memory error)

  enum e_block_changed_mask
  {
    BLOCK_CHANGED_LEFT   = 1,
    BLOCK_CHANGED_RIGHT  = 2,
    BLOCK_CHANGED_TOP    = 4,
    BLOCK_CHANGED_BOTTOM = 8,
    BLOCK_CHANGED_ANY    = 16
  };

  // Top -> Bottom
  template <bool Forward>
  __device__ int pass_T(const std::uint8_t m[][TILE_SIZE], const std::uint8_t M[][TILE_SIZE],
                        std::uint8_t F[][TILE_SIZE], std::uint32_t D[][TILE_SIZE])
  {
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;
    const int     start_y = Forward ? 1 : TILE_SIZE - 2;
    const int     end_y   = Forward ? TILE_SIZE - 1 : 0;
    const int     x       = threadIdx.x + 1;

    int line_changed = 0;

    for (int y = start_y; y != end_y; y += inc)
    {
      for (int dx = -WINDOW; dx < WINDOW + 1; ++dx)
      {
        const std::uint8_t  q     = clamp(F[y + dy][x + dx], m[y][x], M[y][x]);
        const std::uint32_t new_d = D[y + dy][x + dx] + minus_abs<std::uint32_t>(F[y + dy][x + dx], q);
        if (new_d < D[y][x])
        {
          F[y][x]      = q;
          D[y][x]      = new_d;
          line_changed = true;

          if (y == 1)
            line_changed |= BLOCK_CHANGED_TOP;
          else if (y == TILE_SIZE - 2)
            line_changed |= BLOCK_CHANGED_BOTTOM;
          else
            line_changed |= BLOCK_CHANGED_ANY;
        }
      }
      __syncthreads();
    }

    return line_changed;
  }

  // Left -> Right
  template <bool Forward>
  __device__ int pass(const std::uint8_t m[][TILE_SIZE], const std::uint8_t M[][TILE_SIZE], std::uint8_t F[][TILE_SIZE],
                      std::uint32_t D[][TILE_SIZE])
  {
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;
    const int     start_x = Forward ? 1 : TILE_SIZE - 2;
    const int     end_x   = Forward ? TILE_SIZE - 1 : 0;
    const int     y       = threadIdx.x + 1;

    int line_changed = 0;

    for (int x = start_x; x != end_x; x += inc)
    {
      for (int dy = -WINDOW; dy < WINDOW + 1; ++dy)
      {
        const std::uint8_t  q     = clamp(F[y + dy][x + dx], m[y][x], M[y][x]);
        const std::uint32_t new_d = D[y + dy][x + dx] + minus_abs<std::uint32_t>(F[y + dy][x + dx], q);
        if (new_d < D[y][x])
        {
          F[y][x]      = q;
          D[y][x]      = new_d;
          line_changed = true;

          if (x == 1)
            line_changed |= BLOCK_CHANGED_LEFT;
          else if (x == TILE_SIZE - 2)
            line_changed |= BLOCK_CHANGED_RIGHT;
          else
            line_changed |= BLOCK_CHANGED_ANY;
        }
      }
      __syncthreads();
    }

    return line_changed;
  }
} // namespace dt