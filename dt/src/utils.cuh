#pragma once

#include <dt/image2d_view.hpp>

#include <cstdint>

namespace dt
{
  __global__ void init(const image2d_view<std::uint8_t>& m, image2d_view<std::uint32_t>& D,
                       image2d_view<std::uint8_t>& F);

  __inline__ __device__ int clamp(std::uint8_t v, std::uint8_t m, std::uint8_t M)
  {
    return v < m ? m : (v > M ? M : v);
  }

  template <typename T>
  __inline__ __device__ T minus_abs(T a, T b)
  {
    return a < b ? b - a : a - b;
  }
} // namespace dt