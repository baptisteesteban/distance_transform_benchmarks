#pragma once

#include <dt/image2d_view.hpp>

#include <cstdint>

#include "task_queue.cuh"

namespace dt
{
  // https://github.com/masadcv/FastGeodis/blob/0f6cc1e6e104bee56e6eb9dfdd2cfd29c21a1a3e/FastGeodis/fastgeodis_cuda.cu#L50C1-L53C2
  __forceinline__ __device__ float l1distance_cuda(const float& in1, const float& in2)
  {
    return abs(in1 - in2);
  }

  __global__ void init(const image2d_view<std::uint8_t>& m, image2d_view<std::uint32_t>& D,
                       image2d_view<std::uint8_t>& F);

  __global__ void initialize_task_queue(DeviceTaskQueue q);

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