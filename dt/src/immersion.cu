#include <dt/image2d.hpp>
#include <dt/immersion.hpp>

#include <cassert>
#include <cstdint>
#include <stdexcept>

namespace dt
{
  static constexpr int BLOCK_SIZE = 32;

  __device__ std::uint8_t min(const std::uint8_t a, const std::uint8_t b)
  {
    return a > b ? b : a;
  }

  __device__ std::uint8_t max(const std::uint8_t a, const std::uint8_t b)
  {
    return a < b ? b : a;
  }

  __global__ void immersion_kernel(const std::uint8_t* in, int in_pitch, std::uint8_t* m, std::uint8_t* M,
                                   int out_pitch, int width, int height)
  {
    int x  = blockDim.x * blockIdx.x + threadIdx.x;
    int y  = blockDim.y * blockIdx.y + threadIdx.y;
    int kx = 2 * x;
    int ky = 2 * y;

    if (x < width && y < height)
    {
      m[ky * out_pitch + kx] = in[y * in_pitch + x];
      M[ky * out_pitch + kx] = in[y * in_pitch + x];
      if (x < width - 1)
      {
        m[ky * out_pitch + (kx + 1)] = min(in[y * in_pitch + x], in[y * in_pitch + x + 1]);
        M[ky * out_pitch + (kx + 1)] = max(in[y * in_pitch + x], in[y * in_pitch + x + 1]);
      }
      if (y < height - 1)
      {
        m[(ky + 1) * out_pitch + kx] = min(in[y * in_pitch + x], in[(y + 1) * in_pitch + x]);
        M[(ky + 1) * out_pitch + kx] = max(in[y * in_pitch + x], in[(y + 1) * in_pitch + x]);
      }
      if (x < width - 1 && y < height - 1)
      {
        m[(ky + 1) * out_pitch + kx + 1] = min(min(in[y * in_pitch + x], in[y * in_pitch + x + 1]),
                                               min(in[(y + 1) * in_pitch + x], in[(y + 1) * in_pitch + x + 1]));
        M[(ky + 1) * out_pitch + kx + 1] = max(max(in[y * in_pitch + x], in[y * in_pitch + x + 1]),
                                               max(in[(y + 1) * in_pitch + x], in[(y + 1) * in_pitch + x + 1]));
      }
    }
  }

  void immersion_gpu(const image2d_view<std::uint8_t>& img, image2d_view<std::uint8_t>& m,
                     image2d_view<std::uint8_t>& M)
  {
    assert(img.memory_kind() == e_memory_kind::GPU && img.memory_kind() == m.memory_kind() &&
           img.memory_kind() == M.memory_kind());
    assert(m.width() == 2 * img.width() - 1 && m.height() == 2 * img.height() - 1 && m.width() == M.width() &&
           m.height() == M.height());

    dim3 grid_dim((img.width() / BLOCK_SIZE) + 1, (img.height() / BLOCK_SIZE) + 1);
    dim3 block_dim(BLOCK_SIZE, BLOCK_SIZE);
    immersion_kernel<<<grid_dim, block_dim>>>(img.buffer(), img.pitch(), m.buffer(), M.buffer(), m.pitch(), img.width(),
                                              img.height());
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
      throw std::runtime_error(std::format("Unable to launch immersion cuda kernel: {}", cudaGetErrorString(err)));
    cudaDeviceSynchronize();
  }

  std::pair<image2d<std::uint8_t>, image2d<std::uint8_t>> immersion_gpu(const image2d_view<std::uint8_t>& img)
  {
    assert(img.memory_kind() == e_memory_kind::GPU);
    const int             kwidth  = 2 * img.width() - 1;
    const int             kheight = 2 * img.height() - 1;
    image2d<std::uint8_t> m(kwidth, kheight, e_memory_kind::GPU);
    image2d<std::uint8_t> M(kwidth, kheight, e_memory_kind::GPU);
    immersion_gpu(img, m, M);
    return {m, M};
  }
} // namespace dt