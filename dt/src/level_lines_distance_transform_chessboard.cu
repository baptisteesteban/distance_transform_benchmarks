#include <dt/image2d.hpp>

#include <cassert>
#include <cstdio>
#include <format>

#include "utils.cuh"

namespace dt
{
  static constexpr int BLOCK_SIZE = 16;

  // Left -> Right
  __device__ bool pass(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                       image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D)
  {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int y  = by * BLOCK_SIZE + threadIdx.x;
    if (y == 0 || y >= m.height() - 1)
      return false;

    const int start_x = bx * BLOCK_SIZE + (bx == 0);
    const int end_x   = std::min((bx + 1) * BLOCK_SIZE, m.width() - 1);

    bool line_changed = false;

    for (int x = start_x; x < end_x; x++)
    {
      const auto q     = clamp(F(x - 1, y), m(x, y), M(x, y));
      const auto new_d = D(x - 1, y) + minus_abs<std::uint32_t>(F(x - 1, y), q);
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

    pass(m, M, F, D);
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
      cudaMemset2D(D.buffer(), D.pitch(), 0xFF, D.width() * D.elem_size(), D.height());
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
    while (*changed)
    {
      *changed = false;
      block_propagation<<<grid_dim, block_dim>>>(M, M, F, D, false, changed);
      // block_propagation<<<grid_dim, block_dim>>>(M, M, F, D, false, changed);
      cudaDeviceSynchronize();
      break;
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