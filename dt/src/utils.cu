#include "utils.cuh"

namespace dt
{
  __global__ void initialize_generalised_distance_map(const image2d_view<std::uint8_t>& mask, image2d_view<float>& D,
                                                      float v)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < D.width() && y < D.height())
      D(x, y) = v * (mask(x, y) == 0);
  }
} // namespace dt