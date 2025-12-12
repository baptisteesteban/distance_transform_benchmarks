#include <limits>

#include "utils.cuh"

namespace dt
{
  __global__ void initialize_task_queue(DeviceTaskQueue q)
  {
    const int width  = q.gridDimX;
    const int height = q.gridDimY;
    const int i      = (blockIdx.x * blockDim.x + threadIdx.x);
    const int step   = blockDim.x * gridDim.x;
    const int total  = width * height;

    for (int k = i; k < total; k += step)
    {
      if (q.blockPriorities[k] == 0)
      {
        int bx = k % width;
        int by = k / width;
        q.enqueueTask(bx, by);
      }
    }
  }

  __global__ void initialize_generalised_distance_map(const image2d_view<std::uint8_t>& mask, image2d_view<float>& D,
                                                      float v)
  {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x < D.width() && y < D.height())
      D(x, y) = v * (mask(x, y) == 0);
  }
} // namespace dt