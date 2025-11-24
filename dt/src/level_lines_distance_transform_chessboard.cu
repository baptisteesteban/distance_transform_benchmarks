#include <dt/image2d.hpp>

#include <cuda/atomic>

#include <cassert>
#include <cstdio>
#include <format>

#include "utils.cuh"

namespace dt
{
  static constexpr int BLOCK_SIZE = 16;

  // Top -> Bottom
  template <bool Forward>
  __device__ bool pass_T(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                         image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D)
  {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x  = bx * BLOCK_SIZE + threadIdx.x;
    if (x == 0 || x >= m.width() - 1)
      return false;

    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;
    const int     start_y = Forward ? by * BLOCK_SIZE + (by == 0) : std::min((by + 1) * BLOCK_SIZE - 1, m.height() - 2);
    const int     end_y = Forward ? std::min((by + 1) * BLOCK_SIZE, m.height() - 1) : std::max(by * BLOCK_SIZE - 1, 0);

    bool line_changed = false;

    for (int y = start_y; y != end_y; y += inc)
    {
      const std::uint8_t  q     = clamp(F(x, y + dy), m(x, y), M(x, y));
      const std::uint32_t new_d = D(x, y + dy) + minus_abs<std::uint32_t>(F(x, y + dy), q);
      if (new_d < D(x, y))
      {
        F(x, y)      = q;
        D(x, y)      = new_d;
        line_changed = true;
      }
    }

    return line_changed;
  }

  // Left -> Right
  template <bool Forward>
  __device__ bool pass(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                       image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D)
  {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int y  = by * BLOCK_SIZE + threadIdx.x;
    if (y == 0 || y >= m.height() - 1)
      return false;

    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;
    const int     start_x = Forward ? std::max(1, bx * BLOCK_SIZE) : std::min((bx + 1) * BLOCK_SIZE - 1, m.width() - 2);
    const int     end_x   = Forward ? std::min((bx + 1) * BLOCK_SIZE, m.width() - 1) : std::max(bx * BLOCK_SIZE - 1, 0);

    bool line_changed = false;

    for (int x = start_x; x != end_x; x += inc)
    {
      const std::uint8_t  q     = clamp(F(x + dx, y), m(x, y), M(x, y));
      const std::uint32_t new_d = D(x + dx, y) + minus_abs<std::uint32_t>(F(x + dx, y), q);
      if (new_d < D(x, y))
      {
        F(x, y)      = q;
        D(x, y)      = new_d;
        line_changed = true;
      }
    }

    return line_changed;
  }

  __global__ void block_propagation(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                    image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool even,
                                    bool* changed)
  {
    if (!even && blockIdx.x % 2 == blockIdx.y % 2)
      return;
    if (even && blockIdx.x % 2 != blockIdx.y % 2)
      return;

    __shared__ int block_changed;
    if (threadIdx.x == 0)
      block_changed = 1;
    __syncthreads();

    while (block_changed)
    {
      if (threadIdx.x == 0)
        block_changed = 0;
      __syncthreads();

      int t_changed = 0;
      t_changed += pass<true>(m, M, F, D);
      __syncthreads();
      t_changed += pass<false>(m, M, F, D);
      __syncthreads();
      t_changed += pass_T<true>(m, M, F, D);
      __syncthreads();
      t_changed += pass_T<false>(m, M, F, D);
      __syncthreads();

      if (t_changed)
        atomicOr_block(&block_changed, 1);
      __syncthreads();

      if (threadIdx.x == 0 && block_changed)
        *changed = true;
      __syncthreads();
    }
  }

  void level_lines_distance_transform_chessboard_gpu(const image2d_view<std::uint8_t>& m,
                                                     const image2d_view<std::uint8_t>& M,
                                                     image2d_view<std::uint32_t>&      D)
  {
    assert(m.width() == M.width() && m.height() == M.height() && m.width() == D.width() && m.height() == D.height());
    assert(m.memory_kind() == e_memory_kind::GPU && M.memory_kind() == e_memory_kind::GPU &&
           D.memory_kind() == e_memory_kind::GPU);

    // Initialize the data
    image2d<std::uint8_t> F(m.width(), m.height(), e_memory_kind::GPU);
    {
      dim3 block_dim(32, 32);
      dim3 grid_dim(D.width() / 32 + 1, D.height() / 32 + 1);
      init<<<grid_dim, block_dim>>>(m, D, F);
      cudaDeviceSynchronize();
    }

    bool* changed;
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;

    dim3 grid_dim(m.width() / BLOCK_SIZE + 1, m.height() / BLOCK_SIZE + 1);
    dim3 block_dim(BLOCK_SIZE);
    bool even = true;
    while (*changed)
    {
      *changed = false;
      block_propagation<<<grid_dim, block_dim>>>(m, M, F, D, even, changed);
      cudaDeviceSynchronize();
      even = !even;
    }
    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(
          std::format("Error while running level lines distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<std::uint32_t> level_lines_distance_transform_chessboard_gpu(const image2d_view<std::uint8_t>& m,
                                                                       const image2d_view<std::uint8_t>& M)
  {
    assert(m.width() == M.width() && m.height() == M.height());
    assert(m.memory_kind() == e_memory_kind::GPU && M.memory_kind() == e_memory_kind::GPU);

    image2d<std::uint32_t> D(m.width(), m.height(), e_memory_kind::GPU);
    level_lines_distance_transform_chessboard_gpu(m, M, D);
    return D;
  }
} // namespace dt