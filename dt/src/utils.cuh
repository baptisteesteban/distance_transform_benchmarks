#pragma once

#include <dt/image2d_view.hpp>

#include <cstdint>

#include "task_queue.cuh"

namespace dt
{
  static constexpr int BLOCK_SIZE = 32;
  static constexpr int TILE_SIZE  = BLOCK_SIZE + 2; // Halo of 2 to handle border values

  enum e_block_changed_mask
  {
    BLOCK_CHANGED_LEFT         = 1,
    BLOCK_CHANGED_RIGHT        = 2,
    BLOCK_CHANGED_TOP          = 4,
    BLOCK_CHANGED_BOTTOM       = 8,
    BLOCK_CHANGED_TOP_LEFT     = 16,
    BLOCK_CHANGED_TOP_RIGHT    = 32,
    BLOCK_CHANGED_BOTTOM_LEFT  = 64,
    BLOCK_CHANGED_BOTTOM_RIGHT = 128,
    BLOCK_CHANGED_ANY          = 256
  };

  __global__ void initialize_task_queue(DeviceTaskQueue q);

  __global__ void initialize_generalised_distance_map(const image2d_view<std::uint8_t>& mask, image2d_view<float>& D,
                                                      float v);

  __forceinline__ __device__ std::uint8_t clamp(std::uint8_t v, std::uint8_t m, std::uint8_t M)
  {
    return v < m ? m : (v > M ? M : v);
  }

  template <typename T>
  __forceinline__ __device__ T minus_abs(T a, T b)
  {
    return a < b ? b - a : a - b;
  }
} // namespace dt