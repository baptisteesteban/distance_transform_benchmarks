#include <dt/image2d.hpp>

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
  __global__ void pass(const std::uint8_t* m, const std::uint8_t* M, std::uint8_t* F, std::uint32_t* D, int width,
                       int height, int in_pitch, int out_pitch, bool* changed)
  {
    // Order of traversal metadata
    const int     start_x = Forward ? 1 : width - 2;
    const int     end_x   = Forward ? width - 1 : 0;
    constexpr int inc     = Forward ? 1 : -1;
    constexpr int dx      = -1 * inc;

    // Current line
    const int      y      = blockIdx.x * blockDim.x + threadIdx.x;
    std::uint32_t* D_line = (std::uint32_t*)((std::uint8_t*)D + y * out_pitch);

    if (y == 0 || y == height - 1)
    {
      for (int x = 0; x < width; x += 1)
      {
        D_line[x]           = 0;
        F[y * in_pitch + x] = m[y * in_pitch + x];
      }
    }

    if (y > 0 && y < height - 1)
    {
      D_line[0]                   = 0;
      D_line[width - 1]           = 0;
      F[y * in_pitch]             = m[y * in_pitch];
      F[y * in_pitch + width - 1] = m[y * in_pitch + width - 1];

      for (int x = start_x; x != end_x; x += inc)
      {
        auto d     = D_line[x];
        auto q     = clamp(F[y * in_pitch + x + dx], m[y * in_pitch + x], M[y * in_pitch + x]);
        auto new_d = D_line[x + dx] + minus_abs<std::uint32_t>(F[y * in_pitch + x + dx], q);
        if (new_d < d)
        {
          D_line[x]           = new_d;
          F[y * in_pitch + x] = q;
          *changed            = true;
        }
      }
    }
  }

  // Top -> Bottom pass
  template <bool Forward>
  __global__ void pass_T(const std::uint8_t* m, const std::uint8_t* M, std::uint8_t* F, std::uint32_t* D, int width,
                         int height, int in_pitch, int out_pitch, bool* changed)
  {
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
    while (changed)
    {
      pass<true><<<n_blocks, BLOCK_SIZE>>>(m.buffer(), M.buffer(), F.buffer(), (std::uint32_t*)D.buffer(), m.width(),
                                           m.height(), m.pitch(), D.pitch(), changed);
      pass<false><<<n_blocks, BLOCK_SIZE>>>(m.buffer(), M.buffer(), F.buffer(), (std::uint32_t*)D.buffer(), m.width(),
                                            m.height(), m.pitch(), D.pitch(), changed);
      break;
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