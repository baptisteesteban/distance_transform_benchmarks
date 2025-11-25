#include <dt/image2d.hpp>

#include <cuda/atomic>

#include <cassert>
#include <cstdio>
#include <format>

#include "utils.cuh"

namespace dt
{
  static constexpr int BLOCK_SIZE = 16;
  static constexpr int TILE_SIZE  = BLOCK_SIZE + 2; // Halo of 2 to handle border values

  // Top -> Bottom
  template <bool Forward>
  __device__ bool pass_T(const std::uint8_t m[][TILE_SIZE], const std::uint8_t M[][TILE_SIZE],
                         std::uint8_t F[][TILE_SIZE], std::uint32_t D[][TILE_SIZE])
  {
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;
    const int     start_y = Forward ? 1 : TILE_SIZE - 2;
    const int     end_y   = Forward ? TILE_SIZE - 1 : 0;
    const int     x       = threadIdx.x + 1;

    bool line_changed = false;

    for (int y = start_y; y != end_y; y += inc)
    {
      const std::uint8_t  q     = clamp(F[y + dy][x], m[y][x], M[y][x]);
      const std::uint32_t new_d = D[y + dy][x] + minus_abs<std::uint32_t>(F[y + dy][x], q);
      if (new_d < D[y][x])
      {
        F[y][x]      = q;
        D[y][x]      = new_d;
        line_changed = true;
      }
    }

    return line_changed;
  }

  // Left -> Right
  template <bool Forward>
  __device__ bool pass(const std::uint8_t m[][TILE_SIZE], const std::uint8_t M[][TILE_SIZE],
                       std::uint8_t F[][TILE_SIZE], std::uint32_t D[][TILE_SIZE])
  {
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;
    const int     start_x = Forward ? 1 : TILE_SIZE - 2;
    const int     end_x   = Forward ? TILE_SIZE - 1 : 0;
    const int     y       = threadIdx.x + 1;

    bool line_changed = false;

    for (int x = start_x; x != end_x; x += inc)
    {
      const std::uint8_t  q     = clamp(F[y][x + dx], m[y][x], M[y][x]);
      const std::uint32_t new_d = D[y][x + dx] + minus_abs<std::uint32_t>(F[y][x + dx], q);
      if (new_d < D[y][x])
      {
        F[y][x]      = q;
        D[y][x]      = new_d;
        line_changed = true;
      }
    }

    return line_changed;
  }

  __global__ void block_propagation(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                    image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool even,
                                    bool* changed)
  {
    const int bx    = blockIdx.x;
    const int by    = blockIdx.y;
    const int cur_x = bx * BLOCK_SIZE;
    const int cur_y = by * BLOCK_SIZE + threadIdx.x;

    if (cur_y >= m.height())
      return;

    if (!even && bx % 2 == by % 2)
      return;
    if (even && bx % 2 != by % 2)
      return;

    // Loading tile into shared memory
    __shared__ std::uint8_t s_m[TILE_SIZE][TILE_SIZE];
    __shared__ std::uint8_t s_M[TILE_SIZE][TILE_SIZE];
    __shared__ std::uint8_t s_F[TILE_SIZE][TILE_SIZE];
    __shared__ std::uint32_t s_D[TILE_SIZE][TILE_SIZE];

    {
      const std::uint8_t m0 = m(0, 0);
      for (int _dx = -1; _dx <= BLOCK_SIZE; _dx++)
      {
        const int dx = cur_x + _dx;
        for (int _dy = -BLOCK_SIZE; _dy <= BLOCK_SIZE; _dy += BLOCK_SIZE)
        {
          const int oy = threadIdx.x + _dy;
          if (oy < 0 || oy >= TILE_SIZE)
            continue;

          const int dy     = cur_y + _dy;
          s_m[oy][_dx + 1] = dy >= 0 && dx >= 0 && dx < m.width() && dy < m.height() ? m(dx, dy) : m0;
          s_M[oy][_dx + 1] = dy >= 0 && dx >= 0 && dx < m.width() && dy < m.height() ? M(dx, dy) : m0;
          s_F[oy][_dx + 1] = dy >= 0 && dx >= 0 && dx < m.width() && dy < m.height() ? F(dx, dy) : m0;
          s_D[oy][_dx + 1] = dy >= 0 && dx >= 0 && dx < m.width() && dy < m.height() ? D(dx, dy) : 0;
        }
      }
    }
    __syncthreads();

    // Block level distance transform
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
      t_changed += pass<true>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed += pass<false>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed += pass_T<true>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed += pass_T<false>(s_m, s_M, s_F, s_D);
      __syncthreads();

      if (t_changed)
        atomicOr_block(&block_changed, 1);
      __syncthreads();

      if (threadIdx.x == 0 && block_changed)
        *changed = true;
      __syncthreads();
    }

    // Loading tile into global memory
    for (int _dx = 1; _dx <= BLOCK_SIZE; _dx++)
    {
      const int dx = cur_x + _dx - 1;
      if (dx < m.width())
      {
        D(dx, cur_y) = s_D[threadIdx.x + 1][_dx];
        F(dx, cur_y) = s_F[threadIdx.x + 1][_dx];
      }
    }
    __syncthreads();
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
    bool even    = false;
    int  nrounds = 0;
    while (*changed && nrounds < 1)
    {
      *changed = false;
      block_propagation<<<grid_dim, block_dim>>>(m, M, F, D, even, changed);
      cudaDeviceSynchronize();
      even = !even;
      nrounds += 1;
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