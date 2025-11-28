#include <dt/image2d.hpp>

#include <cuda/atomic>

#include <cassert>
#include <cstdio>
#include <format>

#include "lldt_block_passes.cuh"
#include "utils.cuh"

namespace dt
{
  __global__ static void block_propagation(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                           image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool even,
                                           bool* active, bool* changed)
  {
    const int bx  = blockIdx.x;
    const int by  = blockIdx.y;
    const int bid = by * gridDim.x + bx;

    if (!even && bx % 2 == by % 2)
      return;
    if (even && bx % 2 != by % 2)
      return;
    if (!active[bid])
      return;

    const int x = bx * BLOCK_SIZE;
    const int y = by * BLOCK_SIZE + threadIdx.x;

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
      t_changed |= pass<true>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed |= pass<false>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed |= pass_T<true>(s_m, s_M, s_F, s_D);
      __syncthreads();
      t_changed |= pass_T<false>(s_m, s_M, s_F, s_D);
      __syncthreads();

      if (t_changed)
        atomicOr_block(&block_changed, t_changed);
      __syncthreads();

      if (threadIdx.x == 0 && block_changed)
      {
        if (by > 0 && block_changed & BLOCK_CHANGED_TOP)
          active[(by - 1) * gridDim.x + bx] = true;
        if (by < gridDim.y - 1 && block_changed & BLOCK_CHANGED_BOTTOM)
          active[(by + 1) * gridDim.x + bx] = true;
        if (bx > 0 && block_changed & BLOCK_CHANGED_LEFT)
          active[by * gridDim.x + bx - 1] = true;
        if (bx < gridDim.x - 1 && block_changed & BLOCK_CHANGED_RIGHT)
          active[by * gridDim.x + bx + 1] = true;
        *changed = true;
      }
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
    if (threadIdx.x == 0)
      active[bid] = false;
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
      dim3 grid_dim(D.width() + 31 / 32, D.height() + 31 / 32);
      init<<<grid_dim, block_dim>>>(m, D, F);
      cudaDeviceSynchronize();
    }

    bool* changed;
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;

    const int grid_width  = (m.width() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int grid_height = (m.height() + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3      grid_dim(grid_width, grid_height);
    dim3      block_dim(BLOCK_SIZE);
    bool      even = false;
    bool*     active;
    cudaMalloc(&active, grid_width * grid_height * sizeof(bool));
    cudaMemset(active, 0xFF, grid_width * grid_height * sizeof(bool));
    while (*changed)
    {
      *changed = false;
      block_propagation<<<grid_dim, block_dim>>>(m, M, F, D, even, active, changed);
      cudaDeviceSynchronize();
      even = !even;
    }
    cudaFree(active);
    cudaFree(changed);
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