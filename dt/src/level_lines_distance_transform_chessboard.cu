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
  static constexpr int WINDOW     = 1;              // 0 or 1 (if > 1 memory error)

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
      for (int dx = -WINDOW; dx < WINDOW + 1; ++dx)
      {
        const std::uint8_t  q     = clamp(F[y + dy][x + dx], m[y][x], M[y][x]);
        const std::uint32_t new_d = D[y + dy][x + dx] + minus_abs<std::uint32_t>(F[y + dy][x + dx], q);
        if (new_d < D[y][x])
        {
          F[y][x]      = q;
          D[y][x]      = new_d;
          line_changed = true;
        }
      }
      __syncthreads();
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
      for (int dy = -WINDOW; dy < WINDOW + 1; ++dy)
      {
        const std::uint8_t  q     = clamp(F[y + dy][x + dx], m[y][x], M[y][x]);
        const std::uint32_t new_d = D[y + dy][x + dx] + minus_abs<std::uint32_t>(F[y + dy][x + dx], q);
        if (new_d < D[y][x])
        {
          F[y][x]      = q;
          D[y][x]      = new_d;
          line_changed = true;
        }
      }
      __syncthreads();
    }

    return line_changed;
  }

  __global__ void block_propagation(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                    image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool even,
                                    bool* changed)
  {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int x  = bx * BLOCK_SIZE;
    const int y  = by * BLOCK_SIZE + threadIdx.x;

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
      // Here, x and y denote the start of the block line in memory
      const std::uint8_t m0 = m(0, 0);

      // tx and ty are the tile indices
      for (int tx = 0; tx < TILE_SIZE; ++tx)
      {
        for (int dty = -BLOCK_SIZE; dty <= BLOCK_SIZE; dty += BLOCK_SIZE)
        {
          const int ty = threadIdx.x + dty + 1;
          if (ty < 0 || ty >= TILE_SIZE)
            continue;

          const int gx = x + tx - 1;
          const int gy = y + dty;

          bool valid  = gx >= 0 && gx < m.width() && gy >= 0 && gy < m.height();
          s_m[ty][tx] = valid ? m(gx, gy) : m0;
          s_M[ty][tx] = valid ? M(gx, gy) : m0;
          s_F[ty][tx] = valid ? F(gx, gy) : m0;
          if ((tx == 0 && ty == 0) || (tx == 0 && ty == TILE_SIZE - 1) || (tx == TILE_SIZE - 1 && ty == 0) ||
              (tx == TILE_SIZE - 1 && ty == TILE_SIZE - 1)) // Handling corners
            s_D[ty][tx] = std::numeric_limits<std::int32_t>::max();
          else
            s_D[ty][tx] = valid ? D(gx, gy) : 0;
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
    {
      const int ty = threadIdx.x + 1;
      for (int tx = 0; tx < BLOCK_SIZE; ++tx)
      {
        if (x + tx < m.width() && y < m.height())
        {
          F(x + tx, y) = s_F[ty][tx + 1];
          D(x + tx, y) = s_D[ty][tx + 1];
        }
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
    bool even = false;
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