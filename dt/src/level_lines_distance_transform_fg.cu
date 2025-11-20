#include "dt/image2d_view.hpp"
#include <dt/image2d.hpp>

#include <iostream>

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

  // Left -> right pass
  template <bool Forward>
  __global__ void pass(const image2d_view<std::uint8_t>& m, const image2d_view<std::uint8_t>& M,
                       image2d_view<std::uint8_t>& F, image2d_view<std::uint32_t>& D, bool* changed)
  {
    // Order of traversal metadata
    const int     width   = m.width();
    const int     height  = m.height();
    const int     start_x = Forward ? 1 : width - 2;
    const int     end_x   = Forward ? width - 1 : 0;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;

    (void)width;
    (void)height;
    (void)start_x;
    (void)end_x;
    (void)inc;
    (void)dx;
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
    assert(m.pitch() == F.pitch());
    bool* changed;

    cudaMemset2D(D.buffer(), D.pitch(), 0xFF, D.width(), D.height());
    cudaMallocManaged(&changed, sizeof(bool));
    *changed = true;

    int n_blocks = (D.height() / BLOCK_SIZE) + 1;
    // DEBUG
    int nround = 0;
    while (changed && nround < 100)
    {
      *changed = false;
      pass<true><<<n_blocks, BLOCK_SIZE>>>(m, M, F, D, changed);
      pass<false><<<n_blocks, BLOCK_SIZE>>>(m, M, F, D, changed);
      cudaDeviceSynchronize();
      // break;
      nround++;
      if (nround % 10 == 0)
        std::cout << "Round: " << nround << "\n";
    }
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