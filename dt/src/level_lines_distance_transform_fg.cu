#include <dt/image2d.hpp>

#include <iostream>
#include <stdexcept>

namespace dt
{
  __device__ int clamp(std::uint8_t v, std::uint8_t m, std::uint8_t M)
  {
    return v < m ? m : (v > M ? M : v);
  }

  template <typename T>
  __device__ T minus_abs(T a, T b)
  {
    return a < b ? b - a : a - b;
  }

  __global__ void init(const image2d_view<std::uint8_t>& m, image2d_view<std::uint32_t>& D,
                       image2d_view<std::uint8_t>& F)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < m.width() && y < m.height() && (x == 0 || y == 0 || x == F.width() - 1 || y == F.height() - 1))
    {
      D(x, y) = 0;
      F(x, y) = m(x, y);
    }
  }

  // Left -> Right pass
  template <bool Forward>
  __global__ void pass(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                       image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool* changed)
  {
    // Order of traversal metadata
    const int     width   = m.width();
    const int     start_x = Forward ? 1 : width - 2;
    const int     end_x   = Forward ? width - 1 : 0;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;

    const int y = blockDim.x * blockIdx.x + threadIdx.x;
    if (y < m.height())
    {
      for (int x = start_x; x != end_x; x += inc)
      {
        std::uint8_t  q     = clamp(F(x + dx, y), m(x, y), M(x, y));
        std::uint32_t new_d = D(x + dx, y) + minus_abs(F(x + dx, y), q);
        if (new_d < D(x, y))
        {
          D(x, y)  = new_d;
          F(x, y)  = q;
          *changed = true;
        }
      }
    }
  }

  // Top -> Bottom
  template <bool Forward>
  __global__ void pass_T(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                         image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool* changed)
  {
    const int     height  = m.height();
    const int     start_y = Forward ? 1 : height - 2;
    const int     end_y   = Forward ? height - 1 : 0;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dy      = -1 * inc;

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < m.width())
    {
      for (int y = start_y; y != end_y; y += inc)
      {
        std::uint8_t  q     = clamp(F(x, y + dy), m(x, y), M(x, y));
        std::uint32_t new_d = D(x, y + dy) + minus_abs(F(x, y + dy), q);
        if (new_d < D(x, y))
        {
          D(x, y)  = new_d;
          F(x, y)  = q;
          *changed = true;
        }
      }
    }
  }


  void level_lines_distance_transform_fg_gpu(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                                             image2d_view<std::uint32_t>& D)
  {
    constexpr int BLOCK_SIZE = 128;

    assert(m.width() == M.width() && m.height() == M.height() && m.width() == D.width() && m.height() == D.height() &&
           m.memory_kind() == e_memory_kind::GPU && M.memory_kind() == e_memory_kind::GPU &&
           D.memory_kind() == e_memory_kind::GPU);
    assert(m.pitch() == M.pitch());


    image2d<std::uint8_t> F(m.width(), m.height(), e_memory_kind::GPU);
    bool*                 changed;

    cudaMemset2D(D.buffer(), D.pitch(), 0xFF, D.width() * D.elem_size(), D.height());
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;

    // Initialize algorithm
    {
      dim3 block_dim(32, 32);
      dim3 grid_dim(D.width() / 32 + 1, D.height() / 32 + 1);
      init<<<grid_dim, block_dim>>>(m, D, F);
      cudaDeviceSynchronize();
    }

    // Iteration
    int rounds   = 0;
    int n_blocks = (D.height() / BLOCK_SIZE) + 1;
    while (*changed && rounds < 10000)
    {
      *changed = false;
      pass<true><<<n_blocks, BLOCK_SIZE>>>(m, M, F, D, changed);
      pass<false><<<n_blocks, BLOCK_SIZE>>>(m, M, F, D, changed);
      pass_T<true><<<n_blocks, BLOCK_SIZE>>>(m, M, F, D, changed);
      pass_T<false><<<n_blocks, BLOCK_SIZE>>>(m, M, F, D, changed);
      cudaDeviceSynchronize();
      rounds += 1;
    }

    if (const auto err = cudaGetLastError(); err != cudaSuccess)
      throw std::runtime_error(
          std::format("Error while running level lines distance transform: {}", cudaGetErrorString(err)));
  }

  image2d<std::uint32_t> level_lines_distance_transform_fg_gpu(const image2d_view<std::uint8_t>& m,
                                                               const image2d_view<std::uint8_t>& M)
  {
    assert(m.width() == M.width() && m.height() == M.height() && m.memory_kind() == e_memory_kind::GPU &&
           M.memory_kind() == e_memory_kind::GPU);

    image2d<std::uint32_t> D(m.width(), m.height(), e_memory_kind::GPU);
    level_lines_distance_transform_fg_gpu(m, M, D);
    return D;
  }
} // namespace dt