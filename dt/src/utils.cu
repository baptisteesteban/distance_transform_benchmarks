#include "utils.cuh"

namespace dt
{
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
} // namespace dt