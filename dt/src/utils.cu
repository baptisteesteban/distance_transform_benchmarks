#include <limits>

#include "utils.cuh"

namespace dt
{
  __global__ void init(const image2d_view<std::uint8_t>& m, image2d_view<std::uint32_t>& D,
                       image2d_view<std::uint8_t>& F)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= m.width() || y >= m.height())
      return;

    if (x == 0 || y == 0 || x == F.width() - 1 || y == F.height() - 1)
    {
      D(x, y) = 0;
      F(x, y) = m(x, y);
    }
    else
    {
      D(x, y) = std::numeric_limits<std::int32_t>::max();
    }
  }

  __global__ void initialize_task_queue(DeviceTaskQueue q)
  {
    const int width  = q.gridDimX;
    const int height = q.gridDimY;
    const int i      = (blockIdx.x * blockDim.x + threadIdx.x);
    const int step   = blockDim.x * gridDim.x;

    for (int k = i; 2 * k < width; k += step)
      q.enqueueTask(2 * k, 0);

    for (int k = i, b = (height - 1) % 2; 2 * k + b < width; k += step)
      q.enqueueTask(2 * k + b, height - 1);

    for (int k = i; 2 * k < height; k += step)
      q.enqueueTask(0, 2 * k);

    for (int k = i, b = (width - 1) % 2; 2 * k + b < height; k += step)
      q.enqueueTask(width - 1, 2 * k + b);
  }

  __global__ void initialize_geodesic_distance_map(const image2d_view<std::uint8_t>& mask, image2d_view<float>& D,
                                                   float v)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < D.width() && y < D.height())
      D(x, y) = v * (mask(x, y) > 0);
  }
} // namespace dt